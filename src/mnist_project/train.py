import matplotlib.pyplot as plt
import torch
import typer
from mnist_project.data import corrupt_mnist
from mnist_project.model import MyAwesomeModel
import hydra
import typer.completion
from hydra import compose, initialize
# from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score
# from time import time

import wandb

app = typer.Typer()
# @hydra.main(config_path="config", config_name="config.yaml")


@app.command()
def train() -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    with initialize(config_path="../../config"):
        cfg = compose(config_name="config.yaml")

    hparams = cfg.training
    model = MyAwesomeModel().to(DEVICE)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    print("lr = {}, batch_size = {}, epochs = {}".format(cfg.optimizer['lr'], hparams['batch_size'], hparams['epochs']))

    run = wandb.init(
        entity="Seb_Jones",
        project="corrupt_mnist",
        config={"lr": cfg.optimizer['lr'], "batch_size": hparams['batch_size'], "epochs": hparams['epochs']},
        name="run",
    )

    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=hparams['batch_size'])

    loss_fn = torch.nn.CrossEntropyLoss()

    statistics = {"train_loss": [], "train_accuracy": []}
    # wandb.log(statistics)
    for epoch in range(hparams['epochs']):
        model.train()

        preds, targets = [], []
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

                # add a plot of the input images
                images = wandb.Image(img[:5].detach().cpu(), caption="Input images")
                wandb.log({"images": images})

                # add a plot of histogram of the gradients
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                wandb.log({"gradients": wandb.Histogram(grads)})

        # add a custom matplotlib plot of the ROC curves
        preds = torch.cat(preds, 0)
        targets = torch.cat(targets, 0)

        wandb.log({"roc": wandb.plot.roc_curve(targets, preds,
                                               labels=None, classes_to_plot=None)})

        # for class_id in range(10):
        #     one_hot = torch.zeros_like(targets)
        #     one_hot[targets == class_id] = 1
        #     _ = RocCurveDisplay.from_predictions(
        #         one_hot,
        #         preds[:, class_id],
        #         name=f"ROC curve for {class_id}",
        #         plot_chance_level=(class_id == 2),
        #     )

        # wandb.plot({"roc": plt})

    print("Training complete")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(hparams['fig_path'])

    final_accuracy = accuracy_score(targets, preds.argmax(dim=1))
    final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted")
    final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted")
    final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")

    # first we save the model to a file then log it as an artifact
    torch.save(model.state_dict(), hparams['model_path'])
    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
        metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    )
    artifact.add_file(hparams['model_path'])
    run.log_artifact(artifact)


if __name__ == "__main__":
    typer.run(train)
