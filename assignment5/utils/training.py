import os
import torch
import copy as cp
import numpy as np
from torch import nn
from typing import Union, Tuple, Optional, List, Dict, Any
from sklearn.metrics import  balanced_accuracy_score, classification_report, confusion_matrix
from transformers import Trainer, TrainerCallback, TrainerState, TrainerControl, EvalPrediction

from .dml_utils import attach_device


# ------------------------- helpers -------------------------
def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _infer_class_names(trainer: Trainer, n_classes: int) -> List[str]:
    ds = trainer.eval_dataset or trainer.train_dataset
    names = None
    if hasattr(ds, "classes"):
        if isinstance(ds.classes, dict):
            names = [ds.classes.get(i, f"c{i}") for i in range(n_classes)]
        elif isinstance(ds.classes, (list, tuple)):
            names = list(ds.classes)
    if not names or len(names) != n_classes:
        names = [f"c{i}" for i in range(n_classes)]
    return names


# ------------------------- main trainer -------------------------
class DirectMLTrainer(Trainer):
    """
    Logical fixes:
      • Model never computes loss; trainer always computes manual CE (train & eval).
      • Single source of truth for class weights & label smoothing.
      • DirectML/privateuseone placement & safe CPU serialization preserved.
      • compute_metrics returns scalars; rich W&B artifacts logged as side-effects.
    """
    def __init__(self, *args, class_weights=None, dml_device=None, dml_active=False, **kwargs):
        self.dml_device = dml_device
        self.dml_active = dml_active
        self.class_weights = class_weights
        self._opt_built = False

        # wire a default compute_metrics if caller didn't provide one
        if not kwargs.get("compute_metrics", None):
            kwargs["compute_metrics"] = self.compute_metrics

        super().__init__(*args, **kwargs)

        # reflect the requested device for HF internals
        try:
            self.args._device = dml_device
        except Exception:
            pass

        # pin accelerator to target device (no extra wrapping)
        self.accelerator = attach_device(self.accelerator, dml_device)

        # Source of truth for classes & weights
        self.num_classes = int(getattr(getattr(self.model, "config", None), "num_labels", 0) or 0)
        if self.num_classes <= 1:
            raise ValueError("DirectMLTrainer: model.config.num_labels must be > 1 for classification.")

        # class weights live on the model (if provided during model init)
        self.class_weights = getattr(self.model, "class_weights", None)

    # ---- placement/wrapping overrides ----
    def _move_model_to_device(self, model, device):
        return  # we place once in create_optimizer

    def _wrap_model(self, model, training=True, dataloader=None):
        return model  # keep params stable for optimizer

    def log(self, logs: Dict[str, Any], start_time: Optional[float] = None, pop_loss: bool = True) -> None:
        logs = cp.deepcopy(logs)

        if "loss" in logs.keys() and pop_loss:
            logs.pop("loss", None)

        return super().log(logs, start_time)

    def create_optimizer(self):
        if self._opt_built and self.optimizer is not None:
            return
        self.model.to(self.dml_device)

        if self.optimizer is not None:
            self._opt_built = True
            return

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.args.weight_decay,
        )
        self._opt_built = True

    # ---- inputs/tensors ----
    def _prepare_inputs(self, inputs):
        dev = next(self.model.parameters()).device

        def to_dev(x):
            return x.to(dev) if isinstance(x, torch.Tensor) else x

        batch = {k: to_dev(v) for k, v in inputs.items()}
        if "labels" in batch and isinstance(batch["labels"], torch.Tensor):
            batch["labels"] = batch["labels"].long()  # class indices
        return batch

    def _get_logits(self, outputs):
        return outputs.logits if hasattr(outputs, "logits") else outputs[0]

    def compute_loss(self, model, inputs, return_outputs=False, **_):
        labels = inputs["labels"]
        outputs = model(**{ k: v for k, v in inputs.items() if k != "labels" })

        logits = outputs.logits

        weight = self.class_weights.to(logits.device) if isinstance(self.class_weights, torch.Tensor) else None

        ls = float(getattr(self.args, "label_smoothing_factor", 0.0) or 0.0)
        loss = torch.nn.functional.cross_entropy(logits, labels.long(), weight=weight, label_smoothing=ls if ls > 0 else 0.0)

        return (loss, outputs) if return_outputs else loss

    # ---- prediction step returning (loss, logits, labels) ----
    @torch.no_grad()
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels", None)

        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = self._get_logits(outputs)

        loss = None
        if labels is not None:
            weight = self.class_weights.to(logits.device) if isinstance(self.class_weights, torch.Tensor) else None
            label_smoothing = (
                float(getattr(self.args, "label_smoothing_factor", 0.0))
                if getattr(self.args, "label_smoothing_factor", None) is not None
                else float(getattr(self.model, "label_smoothing", 0.0) or 0.0)
            )
            loss = torch.nn.functional.cross_entropy(
                logits, labels.long(),
                weight=weight,
                label_smoothing=label_smoothing if label_smoothing > 0 else 0.0,
                reduction="mean",
            )

        if prediction_loss_only:
            return loss, None, None

        return loss, logits.detach(), labels.detach() if labels is not None else None

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = {k: v.detach().to("cpu") for k, v in self.model.state_dict().items()}
        else:
            state_dict = {k: v.detach().to("cpu") if hasattr(v, "device") else v for k, v in state_dict.items()}
        self.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

    def compute_metrics(self, eval_preds: Union[EvalPrediction, Tuple[np.ndarray, np.ndarray]]):
        if isinstance(eval_preds, EvalPrediction):
            logits = _to_np(eval_preds.predictions)
            labels = _to_np(eval_preds.label_ids)
        else:
            logits, labels = eval_preds
            logits, labels = _to_np(logits), _to_np(labels)

        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        preds = logits.argmax(-1).astype(int)
        labels = labels.astype(int)
        bal_acc = balanced_accuracy_score(labels, preds)

        class_names = _infer_class_names(self, self.num_classes)
        report = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0.0)
        cm = confusion_matrix(labels, preds, labels=list(range(self.num_classes))).tolist()

        return {
            "balanced_accuracy": float(bal_acc),
            "classification_report": report,
            "confusion_matrix_table": cm
        }


# ------------------------- ONLY trigger train-eval -------------------------
class TrainEvalCallback(TrainerCallback):
    """
    Only responsibility: whenever eval runs, also evaluate on the training split
    and let HF/compute_metrics handle all logging/metrics.
    """
    def __init__(self, trainer: DirectMLTrainer):
        self.trainer = trainer

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics and any(k.startswith("eval") for k in metrics.keys()):
            results = self.trainer.evaluate(
                eval_dataset=self.trainer.train_dataset,
                metric_key_prefix="train"
            )

            tl = results.get("train_loss", None)

            self.trainer.log({ "loss": tl }, pop_loss=False)

        return control
