"""Training a Movie Model"""

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torchvision import datasets
import torchvision.transforms as transforms

from datetime import datetime

from ml.model import PetClassifier

from config import BATCH_SIZE, img_size

TRAINED_MODEL_DIR = "trained_models/"
LOG_DATA_DIR = "runs/"

# batch_size - how many samples per batch to load
EPOCHS = 190

# Optimization
LEARNING_RATE = 0.00000001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001


def train_one_epoch(
    epoch_index: int,
    tb_writer: SummaryWriter,
    training_loader: DataLoader,
    optimizer: SGD,
    device: str,
    model: PetClassifier,
    loss_fn: CrossEntropyLoss,
):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(training_loader):
        inputs, labels = data

        # Zero the gradients for every batch!
        optimizer.zero_grad()

        # Perform a transform on the data for it to be usable for the model
        inputs = inputs.to(device)
        inputs = inputs.type(torch.float32)

        labels = labels.to(device)

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        if i % 10 == 0:
            last_loss = running_loss / 10  # loss per batch
            print(f"batch {i + 1} loss: {last_loss}")
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


def main():
    dataset = datasets.OxfordIIITPet(
        root="dataset",
        split="trainval",
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(img_size),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )

    trg_per = 0.85
    val_per = 1.00 - trg_per
    training_size = int(len(dataset) * trg_per)
    validation_size = int(len(dataset) * val_per)

    training_set, validation_set = torch.utils.data.random_split(
        dataset, [training_size, validation_size]
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Using {device} device")
    print("")

    pin_memory = False
    num_workers = 8

    match device:
        case "cpu":
            # Intel(R) Core(TM) i7-4770
            # 4 cores
            # 2 threads per core
            # 8 threads in total
            cores = 4
            threads = 2
            num_workers = cores * threads / 4
        case "cuda":
            # NVIDIA GeForce GTX 1060 6GB
            # 1280 CUDA cores
            # 32 threads per core
            # 40960 threads in total
            pin_memory = True
        case "mps":
            # Macbook Air (M1, 2020)
            # 7 cores
            # 16 threads per core
            pin_memory = True

    training_loader = DataLoader(
        training_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Report the sizes of the datasets
    print(f"Training set has {len(training_set)} instances")
    print(f"Validation set has {len(validation_set)} instances")
    print("")

    # Model
    model = PetClassifier().to(device)

    # Loss Function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Stochastic gradient descent optimization algorithm
    # 1. Increase the momentum from zero to:
    #   1. accelerate convergence
    #   2. smooth out the oscillations
    # 2. Enable Nesterov Momentum to improve the convergence
    # speed of stochastic gradient descent.
    # 3. Increase the weight decay from zero to:
    #   1. prevent overfitting
    #   2. keep the weights small and avoid exploding the gradient
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        nesterov=True,
        weight_decay=WEIGHT_DECAY,
    )

    # Training Loop
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(LOG_DATA_DIR + f"movie_trainer_{timestamp}")
    epoch_number = 0
    best_vloss = 1_000_000.0
    best_accuracy = 0
    best_loss = 1_000_000.0

    for _ in range(EPOCHS):
        print(f"EPOCH {epoch_number + 1}:")

        # 1. Train the Model
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(
            epoch_number,
            writer,
            training_loader,
            optimizer,
            device,
            model,
            loss_fn,
        )

        # 2. Evaluate the Model
        # Set model to evaluation mode
        model.eval()
        size = len(validation_set)
        num_batches = len(validation_loader)
        correct = 0
        running_test_loss = 0.0

        # Disable gradient computation and reduce memory consumption
        with torch.no_grad():
            for _, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata

                vinputs = vinputs.to(device)
                vinputs = vinputs.type(torch.float32)

                vlabels = vlabels.to(device)

                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_test_loss += vloss
                correct += (
                    (voutputs.argmax(0) == vlabels)
                    .type(torch.float)
                    .sum()
                    .item()
                )

        avg_vloss = running_test_loss / num_batches
        accuracy = 100 * (correct / size)
        print(f"Accuracy: {accuracy}%")
        print(f"Training loss: {avg_loss}, Validation loss: {avg_vloss}")

        # Log the running loss average per batch
        # for both training and testing
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss},
            epoch_number + 1,
        )
        # Log the accuracy per batch
        writer.add_scalars(
            "Accuracy",
            {"Accuracy": accuracy},
            epoch_number + 1,
        )
        writer.flush()

        # Track the best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            # Save the model's state
            model_path = (
                TRAINED_MODEL_DIR + f"model_{timestamp}_{epoch_number}.pt"
            )
            torch.save(model.state_dict(), model_path)

        if accuracy > best_accuracy:
            best_accuracy = accuracy

        if avg_loss < best_loss:
            best_loss = avg_loss

        epoch_number += 1

    print("")
    print(f"Best Accuracy: {best_accuracy}%")
    print(
        f"Lowest Training loss: {best_loss}, "
        f"Lowest Validation loss: {best_vloss}"
    )


if __name__ == "__main__":
    main()
