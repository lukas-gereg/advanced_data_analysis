import os
import torch
import wandb
import random
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback

from utils.training import DirectMLTrainer, TrainEvalCallback
from dataloaders.reuters_dataset import load_reuters_texts, ReutersDS
from utils.dml_utils import AnyDeviceTrainingArguments, can_allocate_on
from models.reuters_transformer_model import CustomTransformerConfig, CustomTransformerForSequenceClassification


EARLY_STOPPING = 4
SEED = 1337
MAX_LEN = 256
BATCH_SIZE = 64
EPOCHS = 24
LR = 1e-3

EMBED_DIM = 256
N_LAYERS = 6
N_HEADS = 8
D_FF = 1024
DROPOUT = 0.2


def set_seed_all(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_directml_device():
    """
    Prefer privateuseone:0; fallback to torch_directml; else CPU.
    """

    try:
        import torch_directml
        dml = torch_directml.device()
        if can_allocate_on(dml):
            return dml, True, "directml"
    except Exception:
        pass

    # 3) CPU
    return torch.device("cpu"), False, "cpu"


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed_all(SEED)

    dml_device, dml_active, device_label = get_directml_device()

    wandb.login(key="a9f105e8b3bc98e07700e93201d4b02c1c75106d")
    wandb.init(
        project="PMAD", entity="DP_Gereg",
        config={
            "seed": SEED,
            "embedding_dim": EMBED_DIM,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "d_ff": D_FF,
            "dropout": DROPOUT,
            "max_len": MAX_LEN,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
            "device": device_label,
        },
    )

    # Data
    (tr_texts, tr_y), (te_texts, te_y), num_labels, class_weights = load_reuters_texts(num_words=20_000, test_split=0.2)

    # Make a validation split from training data
    tr_texts, val_texts, tr_y, val_y = train_test_split(
        tr_texts, tr_y, test_size=0.15, random_state=SEED, stratify=tr_y
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    train_ds = ReutersDS(tr_texts, tr_y, tokenizer, max_len=MAX_LEN)
    val_ds   = ReutersDS(val_texts, val_y, tokenizer, max_len=MAX_LEN)
    test_ds  = ReutersDS(te_texts, te_y, tokenizer, max_len=MAX_LEN)

    # Model
    cfg = CustomTransformerConfig(
        vocab_size=tokenizer.vocab_size or 30_522,
        num_labels=num_labels,
        d_model=EMBED_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        max_position_embeddings=MAX_LEN,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = CustomTransformerForSequenceClassification(cfg)

    args = AnyDeviceTrainingArguments(
        remove_unused_columns=False,
        device=dml_device,
        output_dir="./out_reuters_directml",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        num_train_epochs=EPOCHS,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=False,
        report_to=["wandb"],
        use_cpu=True,
        label_smoothing_factor=0.1,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = DirectMLTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        processing_class=tokenizer,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dml_device=dml_device,
        dml_active=dml_active,
    )

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING))
    trainer.add_callback(TrainEvalCallback(trainer))

    trainer.train()

    final_test = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    trainer.log(final_test)

    wandb.finish()


if __name__ == "__main__":
    main()
