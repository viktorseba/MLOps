import wandb
import torch
from model import MyAwesomeModel

run = wandb.init()
artifact = run.use_artifact('jonasschitt-danmarks-tekniske-universitet-dtu/corrupt_mnist/corrupt_mnist_model:v0', type='model')
artifact_dir = artifact.download()
model = MyAwesomeModel()
model.load_state_dict(torch.load("artifact_dir/model.ckpt"))
