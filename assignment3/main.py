import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
import torch
from torch.utils.data import DataLoader
import torch_directml
import wandb
from sklearn.preprocessing import StandardScaler

from assignments.assignment3.utils.evaluation import Evaluation
from assignments.assignment3.utils.training import Training
from dataloaders.csvloader import CsvLoader
from models.sonar_model import SonarModel

def initialize(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)

LR = 0.001
BATCH_SIZE = 10
EARLY_STOPPING = 5
EPOCHS = 100


data = pd.read_csv("./data/sonar.all-data", sep=",", header=None)
data[data.columns[-1]] = data.iloc[:, -1].apply(lambda value: 1 if value == "R" else 0)

train, test = train_test_split(data, test_size=0.3, random_state=42, stratify=data.iloc[:, -1])
test, validation = train_test_split(test, test_size=0.3, random_state=42, stratify=test.iloc[:, -1])

classes = { 1: "R", 0: "M" }

scaler = StandardScaler()

train[train.columns[:-1]] = scaler.fit_transform(train.iloc[:, :-1])
test[test.columns[:-1]] = scaler.transform(test.iloc[:, :-1])
validation[validation.columns[:-1]] = scaler.transform(validation.iloc[:, :-1])

train_set = CsvLoader(train).set_classes(classes)
test_set = CsvLoader(test).set_classes(classes)
validation_set = CsvLoader(validation).set_classes(classes)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False)

models = [
    SonarModel({ "size": [20, 1], "input_size": train_set.x.shape[1] }),
    # SonarModel({ "size": [64, 128, 32, 1], "input_size": train_set.x.shape[1] }),
]

optimizers = [
    # torch.optim.Adam,
    # torch.optim.SGD,
    # torch.optim.Adagrad,
    torch.optim.RMSprop
]

loss = nn.BCELoss()
# device = torch_directml.device()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
wandb_config = dict(project="PMAD", entity="DP_Gereg", config={
        "learning rate": LR,
        "epochs": EPOCHS,
        "early stopping": EARLY_STOPPING,
        "loss calculator": str(loss),
        "batch_size": BATCH_SIZE,
    })

wandb.login(key="a9f105e8b3bc98e07700e93201d4b02c1c75106d")

for model in models:
    for optimizer in optimizers:
        wandb_config["config"]["model"] = str(model)
        wandb_config["config"]["optimizer"] = str(optimizer)
        wandb_config["config"]["model_properties"] = model.defaults

        model.classifier.apply(initialize)

        wandb.init(**wandb_config)
        optim = optimizer(model.parameters(), lr=LR)

        Training()(EPOCHS, device, optim, model, loss, train_loader, validation_loader, EARLY_STOPPING)
        Evaluation()(loss, test_loader, model, device)

        if wandb.run is not None:
            wandb.finish()



