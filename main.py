# -*- coding: utf-8 -*-
import os
import pickle
import warnings

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from architectures import DepixelationCNN
from datasets import RandomImagePixelationDataset, ImageDepixelationTestDataset
from submission_serialization import serialize

from utils import plot, stack_with_padding


def evaluate_model(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        loss_fn,
        device: torch.device,
):
    model.eval()
    # We will accumulate the mean loss
    loss = 0
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in the specified data loader
        for data in tqdm(loader, desc="Evaluating", position=0, leave=False):
            # Get a sample and move inputs and targets to device
            inputs, known_arrays, targets, _ = data
            inputs = inputs.to(device)
            targets = [target.to(device) for target in targets]

            # Get outputs of the specified model
            outputs = model(inputs)

            # Compute loss based on output and targets
            losses = []
            for i in range(len(outputs)):
                output = outputs[i]
                target = targets[i]
                known_array = known_arrays[i]

                masked_output = output[(~known_array).bool()]
                target_flatten = target.flatten()

                loss = loss_fn(masked_output, target_flatten)
                losses.append(loss)

            loss = sum(losses) / len(losses)

    loss /= len(loader)
    model.train()
    return loss


def main(
        results_path,
        network_config: dict,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 15,
        device: str = "cuda",
):
    """Main function that takes hyperparameters and performs training and
    evaluation of model"""
    device = torch.device(device)
    if "cuda" in device.type and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    # Set a known random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Prepare a path to plot to
    plot_path = os.path.join(results_path, "plots")
    os.makedirs(plot_path, exist_ok=True)

    # Load training dataset

    transformations = transforms.Compose(
        [transforms.Resize(size=64), transforms.CenterCrop(size=(64, 64))]
    )
    dataset = RandomImagePixelationDataset(
        image_dir="./data/training",
        width_range=(4, 32),
        height_range=(4, 32),
        size_range=(4, 16),  # transform= transformations
    )

    # Split dataset into training, validation and test set (CIFAR10 dataset
    # is already randomized, so we do not necessarily have to shuffle again)
    training_set = torch.utils.data.Subset(
        dataset, indices=np.arange((int(len(dataset) * (3 / 5))))
    )
    validation_set = torch.utils.data.Subset(
        dataset, indices=np.arange(int(len(dataset) * (3 / 5)), len(dataset))
    )

    train_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=stack_with_padding,
        pin_memory=True,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=1, shuffle=False, num_workers=4
    )

    # Define a TensorBoard summary writer that writes to directory
    # "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))

    # Create Network
    net = DepixelationCNN(**network_config)
    net.to(device)
    net.train()
    # Get mse loss function
    mse = torch.nn.MSELoss()

    # Get adam optimizer
    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    write_stats_at = 1000  # Write status to TensorBoard every x updates
    plot_at = 10_000  # Plot every x updates
    validate_at = 5000  # Evaluate model on validation set and check for new best model every x updates
    update = 0  # Current update counter
    best_validation_loss = np.inf  # Best validation loss so far

    # Save initial model as "best" model (will be overwritten later)
    saved_model_file = os.path.join(results_path, "depixelation_cnn.pt")
    torch.save(net, saved_model_file)

    # Training loop
    for epoch in range(epochs):
        train_loader = tqdm(train_loader, desc=f"Epoch: {epoch + 1}", position=0)
        for data in train_loader:
            # Get next samples
            inputs, known_arrays, targets, _ = data

            inputs = inputs.to(torch.device("cuda"))

            known_arrays = known_arrays.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Get outputs of our network
            outputs = net(inputs)
            targets = [target.to(device) for target in targets]
            losses = []
            for i in range(len(outputs)):
                output = outputs[i]
                target = targets[i]
                known_array = known_arrays[i]

                masked_output = output[(~known_array).bool()]
                target_flatten = target.flatten()

                loss = mse(masked_output, target_flatten)
                losses.append(loss)

            loss = sum(losses) / len(losses)

            loss.backward()
            optimizer.step()

            # Write current training status
            if (update + 1) % write_stats_at == 0:
                writer.add_scalar(
                    tag="Loss/training", scalar_value=loss.cpu(), global_step=update
                )
                for i, (name, param) in enumerate(net.named_parameters()):
                    writer.add_histogram(
                        tag=f"Parameters/[{i}] {name}",
                        values=param.cpu(),
                        global_step=update,
                    )
                    writer.add_histogram(
                        tag=f"Gradients/[{i}] {name}",
                        values=param.grad.cpu(),
                        global_step=update,
                    )

            # Plot output
            if (update + 1) % plot_at == 0:
                plot(
                    inputs.detach().cpu().numpy(),
                    targets.detach().cpu().numpy(),
                    outputs.detach().cpu().numpy(),
                    plot_path,
                    update,
                )

            # Evaluate model on validation set
            if (update + 1) % validate_at == 0:
                val_loss = evaluate_model(
                    net, loader=validation_loader, loss_fn=mse, device=device
                )
                writer.add_scalar(
                    tag="Loss/validation", scalar_value=val_loss, global_step=update
                )
                # Save best model for early stopping
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    torch.save(net, saved_model_file)

            train_loader.set_description(f"loss: {loss:7.5f}", refresh=True)
            train_loader.update()

            # Increment update counter
            update += 1

    train_loader.close()
    writer.close()
    print("Finished Training!")
    torch.save(net, saved_model_file)
    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(saved_model_file)
    train_loss = evaluate_model(net, loader=train_loader, loss_fn=mse, device=device)
    val_loss = evaluate_model(net, loader=validation_loader, loss_fn=mse, device=device)
    # test_loss = evaluate_model(net, loader=test_loader, loss_fn=mse, device=device)

    print(f"Scores:")
    print(f"training loss: {train_loss}")
    print(f"validation loss: {val_loss}")
    # print(f"test loss: {test_loss}")

    # Write result to file
    with open(os.path.join(results_path, "results.txt"), "w") as rf:
        print(f"Scores:", file=rf)
        print(f"training loss: {train_loss}", file=rf)
        print(f"validation loss: {val_loss}", file=rf)
        # print(f"test loss: {test_loss}", file=rf)

    test_set = ImageDepixelationTestDataset("./data/test_set.pkl")
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=0
    )

    # To store predictions
    predictions = []

    # Evaluate the model
    with torch.no_grad():
        for (
                stacked_pixelated_images,
                stacked_known_arrays,
        ) in test_loader:
            stacked_pixelated_images = stacked_pixelated_images.to(device)
            stacked_known_arrays = stacked_known_arrays.to(device)

            outputs = net(stacked_pixelated_images)

            for i in range(len(outputs)):
                output = outputs[i]
                known_array = stacked_known_arrays[i]

                # Get the output where the known array is False, i.e., the pixelated images
                predicted_output = output[~known_array.bool()]

                predicted_output = predicted_output * 255

                predicted_output = (
                    predicted_output.detach().cpu().numpy().astype(np.uint8)
                )

                # Append flattened array to the list
                predictions.append(predicted_output.flatten())

    serialize(predictions, "predictions.bin")


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to JSON config file.")
    args = parser.parse_args()
    with open(args.config_file) as cf:
        config = json.load(cf)

    main(**config)
