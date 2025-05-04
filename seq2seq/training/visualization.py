"""
Visualization utilities for training results.
"""

import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_losses(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show_plot: bool = False,
) -> None:
    """
    Plot training and validation losses.

    Args:
        history: Dictionary containing training and validation losses
        save_path: Path to save the plot image
        show_plot: Whether to display the plot interactively
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot train and validation loss
    plt.plot(epochs, history["train_loss"], "bo-", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "ro-", label="Validation Loss")

    # Add epoch numbers to x-axis
    plt.xticks(epochs)

    # Add grid and styling
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Print losses to console
    print("\nLoss per epoch:")
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<12}")
    print(f"{'-' * 6}+{'-' * 14}+{'-' * 14}")
    for i, (train_loss, val_loss) in enumerate(zip(history["train_loss"], history["val_loss"])):
        print(f"{i + 1:<6} | {train_loss:<12.4f} | {val_loss:<12.4f}")

    # Save the plot
    if save_path:
        plt.savefig(save_path)
        print(f"\nLoss plot saved to: {save_path}")

    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
