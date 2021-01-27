import torch
from ignite.distributed import auto_dataloader, auto_model
from ignite.metrics import Accuracy, Precision, Recall
from ignite.utils import to_onehot
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR

from matches.accelerators import DDPAccelerator
from matches.callbacks import BestModelSaver, TqdmProgressCallback
from matches.callbacks.tensorboard import TensorboardMetricWriterCallback
from matches.loop import Loop
from matches.utils import seed_everything, setup_cudnn_reproducibility, unique_logdir
from .utils import get_model, get_train_test_datasets

NUM_EPOCHS = 20

def run(loop: Loop):
    seed_everything(42)
    setup_cudnn_reproducibility(True, False)

    train_ds, valid_ds = get_train_test_datasets("data/cifar")

    model = auto_model(get_model())

    train_loader = auto_dataloader(
        train_ds,
        batch_size=512,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    valid_loader = auto_dataloader(
        valid_ds,
        batch_size=512,
        num_workers=4,
        shuffle=False,
    )

    optim = SGD(model.parameters(), lr=0.4, momentum=0.9)

    scheduler = OneCycleLR(
        optim, max_lr=1, epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader)
    )
    criterion = CrossEntropyLoss()

    precision = Precision(average=False)
    recall = Recall(average=False)

    # Ignite metrics are combinable
    f1 = (precision * recall * 2 / (precision + recall)).mean()
    accuracy = Accuracy()

    # We are attaching metrics to automatically reset
    loop.attach(
        # Loop manages train/eval modes, device and requires_grad of attached `nn.Module`s
        criterion=criterion,
        # This criterion doesn't have any state or attribute tensors
        # So it's attachment doesn't introduce any behavior
        model=model,
        # Loop saves state of all attached objects having state_dict()/load_state_dict() methods
        # to checkpoints
        optimizer=optim,
        scheduler=scheduler,
    )

    def train(loop: Loop):
        for _ in loop.iterate_epochs(NUM_EPOCHS):
            for x, y in loop.iterate_dataloader(train_loader, mode="train"):
                y_pred_logits = model(x)

                loss: torch.Tensor = criterion(y_pred_logits, y)
                loop.backward(loss)
                loop.optimizer_step(optim)
                scheduler.step()
                loop.metrics.log("lr", scheduler.get_last_lr()[0])

            for x, y in loop.iterate_dataloader(valid_loader):
                y_pred_logits: torch.Tensor = model(x)

                y_pred = to_onehot(y_pred_logits.argmax(dim=-1), num_classes=10)

                precision.update((y_pred, y))
                recall.update((y_pred, y))
                accuracy.update((y_pred, y))

            loop.metrics.log("valid/precision", precision.compute().mean())
            loop.metrics.log("valid/recall", recall.compute().mean())
            loop.metrics.consume("valid/f1", f1)
            loop.metrics.consume("valid/accuracy", accuracy)

    loop.run(train)


loop = Loop(
    unique_logdir("logs/cifar"),
    [
        BestModelSaver("valid/f1", metric_mode="max"),
        TqdmProgressCallback(),
        TensorboardMetricWriterCallback()
    ],
)

if __name__ == "__main__":
    loop.launch(run, DDPAccelerator("2"))
