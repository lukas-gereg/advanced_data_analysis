import os
import torch
import wandb
import random
import string
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, matthews_corrcoef, jaccard_score

from .optimizer_rebinder import ensure_optimizer_matches_model
from .validation import Validation


class Training:
    def __init__(self, debug: bool = False):
        self.validation = Validation(debug)
        self.debug = debug
        self.run_name = ""

    def __call__(self, epochs, device, optimizer, model, loss, train_loader, validation_loader, threshold, validation_scheduler=None):
        if wandb.run is not None:
            self.run_name = wandb.run.name
        else:
            self.run_name = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(15))

        print(f"Training run {self.run_name}")

        model = model.to(device)
        optimizer = ensure_optimizer_matches_model(model, optimizer, preserve_state=True)

        x = torch.tensor([])
        y = torch.tensor([])

        old_validation_value = self.validation(0, validation_loader, device, model, loss)
        counter = 0
        torch.save(obj=model.state_dict(), f=os.path.join("..", "model_params", f"run-{self.run_name}-params.pth"))
        losses = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0

            x = torch.tensor([])
            y = torch.tensor([])

            for batch_idx, (data, labels) in enumerate(train_loader):
                data = data.to(device)
                labels = labels.to(device)

                prediction = model(data)

                current_loss = loss(prediction, labels)

                optimizer.zero_grad()
                current_loss.backward()
                optimizer.step()
                epoch_loss += current_loss.item()

                if self.debug:
                    print(f"training prediction: {prediction.cpu().detach().numpy().tolist()}, class "
                          f"prediction: {np.round(prediction.cpu().detach().numpy()).tolist()}, ground "
                          f"truth: {labels.cpu().detach().numpy().tolist()}")

                x = torch.cat((x, prediction.cpu()))
                y = torch.cat((y, labels.cpu()))

            train_loss = epoch_loss / len(train_loader)

            class_report = classification_report(
                y.detach().numpy(),
                np.round(x.detach().numpy()),
                labels=list(train_loader.dataset.classes.keys()),
                target_names=list(train_loader.dataset.classes.values()),
                output_dict=True,
                zero_division=0.0
            )

            balanced_accuracy = balanced_accuracy_score(y.detach().numpy(), np.round(x.detach().numpy()))

            conf_mat = confusion_matrix(
                y.detach().numpy(),
                np.round(x.detach().numpy()),
                labels=list(train_loader.dataset.classes.keys())
            )

            column_names = list(train_loader.dataset.classes.values())
            conf_mat = conf_mat.tolist()

            print(
                f"Epoch {epoch + 1} train_report: {dict({'balanced_accuracy': balanced_accuracy, **class_report})}, train_loss: {train_loss}, train_confusion_matrix: {conf_mat}")

            if wandb.run is not None:
                for index, row in enumerate(conf_mat):
                    row.insert(0, column_names[index])

                wandb_table = wandb.Table(data=conf_mat, columns=["names (real ↓/predicted →)"] + column_names)

                wandb.log(
                    {f"train_report": {'loss': train_loss, 'balanced_accuracy': balanced_accuracy, **class_report},
                     f"train_confusion_matrix": wandb_table,
                     "epoch": epoch + 1,
                     }, step=epoch + 1)

            current_validation_value = self.validation(epoch + 1, validation_loader, device, model, loss, validation_scheduler)

            losses.append(current_validation_value)

            if threshold is None or current_validation_value < old_validation_value:
                old_validation_value = current_validation_value
                counter = 0
                torch.save(obj=model.state_dict(), f=os.path.join("..", "model_params", f"run-{self.run_name}-params.pth"))
            elif counter < threshold:
                counter += 1
            else:
                model.load_state_dict(torch.load(os.path.join("..", "model_params", f"run-{self.run_name}-params.pth")))
                print(f"Risk of over fitting parameters, ending learning curve at epoch {epoch + 1}, reverting back to epoch {epoch - counter}.")

                print("labels: ", y.detach().numpy().tolist())
                print("predictions: ", x.detach().numpy().tolist())
                print("labels meanings: ", train_loader.dataset.classes)

                return losses[: -counter]

        model.load_state_dict(torch.load(os.path.join("..", "model_params", f"run-{self.run_name}-params.pth")))

        print("labels: ", y.detach().numpy().tolist())
        print("predictions: ", x.detach().numpy().tolist())
        print("labels meanings: ", train_loader.dataset.classes)

        return losses
