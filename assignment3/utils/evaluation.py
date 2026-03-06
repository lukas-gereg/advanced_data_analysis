import torch
import wandb
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, matthews_corrcoef, jaccard_score


class Evaluation:
    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(self, loss, test_loader, model, device):
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            total_loss = 0
            results = []

            x = torch.tensor([])
            y = torch.tensor([])

            for test_loader_data, labels in test_loader:
                test_loader_data = test_loader_data.to(device)
                labels = labels.to(device)

                outputs = model(test_loader_data)
                total_loss += loss(outputs, labels).item()

                x = torch.cat((x, outputs.cpu()))
                y = torch.cat((y, labels.cpu()))

                if self.debug:
                    print(
                        f"evaluation prediction: {outputs.cpu().detach().numpy().tolist()}, "
                        f"class prediction: {np.round(outputs.cpu().detach().numpy()).tolist()}, ground "
                        f"truth: {labels.cpu().detach().numpy().tolist()}")

                results.extend(zip(labels.cpu(), outputs.cpu()))

            eval_loss = total_loss / len(test_loader)

            class_report = classification_report(
                y.detach().numpy(),
                np.round(x.detach().numpy()),
                labels=list(test_loader.dataset.classes.keys()),
                target_names=list(test_loader.dataset.classes.values()),
                output_dict=True,
                zero_division=0.0
            )

            balanced_accuracy = balanced_accuracy_score(y.detach().numpy(), np.round(x.detach().numpy()))

            conf_mat = confusion_matrix(
                y.detach().numpy(),
                np.round(x.detach().numpy()),
                labels=list(test_loader.dataset.classes.keys())
            )

            column_names = list(test_loader.dataset.classes.values())
            conf_mat = conf_mat.tolist()

            print(
                f"evaluation_report: {dict({'balanced_accuracy': balanced_accuracy, **class_report})},"
                f"evaluation_loss: {eval_loss}, evaluation_confusion_matrix: {conf_mat}"
            )

            if wandb.run is not None:
                for index, row in enumerate(conf_mat):
                    row.insert(0, column_names[index])

                wandb_table = wandb.Table(data=conf_mat, columns=["names (real ↓/predicted →)"] + column_names)

                wandb.log(
                    {f"evaluation_report": {'loss': eval_loss, 'balanced_accuracy': balanced_accuracy, **class_report},
                     f"evaluation_confusion_matrix": wandb_table
                     })

            return eval_loss, results
