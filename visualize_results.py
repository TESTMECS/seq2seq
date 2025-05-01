#!/usr/bin/env python
"""
Visualization script for seq2seq translation model results.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def load_loss_history(
    model_path, attn_suffix="_attn_final", no_attn_suffix="_no_attn_final"
):
    """Load loss histories from model files."""
    # Check if history files exist
    attn_path = os.path.join(model_path, f"losses{attn_suffix}.png")
    no_attn_path = os.path.join(model_path, f"losses{no_attn_suffix}.png")

    # Print available models
    print(f"Attention model path: {attn_path}")
    print(f"No attention model path: {no_attn_path}")

    # Since we don't have direct access to the history data, we'll use the existing plots
    # to extract data.
    print(
        "The raw data isn't directly available, but we can visualize the existing plots"
    )

    return None, None


def visualize_model_comparison(save_dir="models", show_plot=True, save_plot=True):
    """Create a comprehensive visualization comparing models with and without attention."""
    # Path for the plots
    save_path = os.path.join(save_dir, "model_comparison.png")

    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

    # Add existing plots as subfigures
    attn_img = plt.imread(os.path.join(save_dir, "losses_attn_final.png"))
    no_attn_img = plt.imread(os.path.join(save_dir, "losses_no_attn_final.png"))

    # Plot images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(no_attn_img)
    ax1.axis("off")
    ax1.set_title("Model without Attention", fontsize=14)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(attn_img)
    ax2.axis("off")
    ax2.set_title("Model with Attention", fontsize=14)

    # Add model architecture comparison
    ax3 = fig.add_subplot(gs[1, :])

    # Model parameters (from the console output)
    no_attn_params = 77604
    attn_params = 133796

    # Create bars for parameter counts
    models = ["Without Attention", "With Attention"]
    params = [no_attn_params, attn_params]

    x = np.arange(len(models))
    bars = ax3.bar(x, params, width=0.6, color=["blue", "red"])

    # Add labels and formatting
    ax3.set_title("Model Complexity Comparison", fontsize=14)
    ax3.set_ylabel("Number of Parameters", fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=12)

    # Add parameter count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 5000,
            f"{height:,}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # Add grid to bottom plot
    ax3.grid(axis="y", linestyle="--", alpha=0.7)

    # Add annotations with architectural differences
    textstr = "\n".join(
        [
            "Key Architectural Differences:",
            "• Attention model uses an attention mechanism to focus on relevant parts of input",
            "• Attention layer adds 56,192 additional parameters (72% increase)",
            "• Decoder in attention model concatenates context vector with embedding",
            "• Final output layer combines hidden state, output, and context for prediction",
        ]
    )

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.4)
    ax3.text(
        0.5,
        0.05,
        textstr,
        transform=ax3.transAxes,
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="center",
        bbox=props,
    )

    # Add overall title
    fig.suptitle("Seq2Seq Translation Model Analysis", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    if save_plot:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Comparison visualization saved to: {save_path}")

    # Show figure
    if show_plot:
        plt.show()
    else:
        plt.close()

    return save_path


def visualize_architecture():
    """Create a visual representation of the model architecture."""
    fig, ax = plt.subplots(figsize=(15, 8))

    # Turn off axes
    ax.axis("off")

    # Draw encoder-decoder architecture
    # This is a simplified visualization of the architecture

    # Define components
    components = [
        {
            "name": "Encoder",
            "x": 0.2,
            "y": 0.5,
            "width": 0.15,
            "height": 0.7,
            "color": "lightblue",
        },
        {
            "name": "Encoder\nEmbedding",
            "x": 0.05,
            "y": 0.5,
            "width": 0.15,
            "height": 0.3,
            "color": "lightyellow",
        },
        {
            "name": "Bidirectional\nGRU",
            "x": 0.2,
            "y": 0.3,
            "width": 0.15,
            "height": 0.4,
            "color": "lightblue",
        },
        {
            "name": "FC",
            "x": 0.2,
            "y": 0.7,
            "width": 0.15,
            "height": 0.1,
            "color": "lightgreen",
        },
        {
            "name": "Attention",
            "x": 0.45,
            "y": 0.5,
            "width": 0.1,
            "height": 0.2,
            "color": "pink",
        },
        {
            "name": "Decoder",
            "x": 0.65,
            "y": 0.5,
            "width": 0.15,
            "height": 0.7,
            "color": "lightblue",
        },
        {
            "name": "Decoder\nEmbedding",
            "x": 0.8,
            "y": 0.5,
            "width": 0.15,
            "height": 0.3,
            "color": "lightyellow",
        },
        {
            "name": "GRU",
            "x": 0.65,
            "y": 0.3,
            "width": 0.15,
            "height": 0.4,
            "color": "lightblue",
        },
        {
            "name": "FC Output",
            "x": 0.65,
            "y": 0.7,
            "width": 0.15,
            "height": 0.1,
            "color": "lightgreen",
        },
    ]

    # Draw components
    for comp in components:
        rect = plt.Rectangle(
            (comp["x"], comp["y"]),
            comp["width"],
            comp["height"],
            fill=True,
            color=comp["color"],
            alpha=0.7,
            linewidth=2,
            edgecolor="black",
        )
        ax.add_patch(rect)
        ax.text(
            comp["x"] + comp["width"] / 2,
            comp["y"] + comp["height"] / 2,
            comp["name"],
            ha="center",
            va="center",
            fontsize=10,
        )

    # Add arrows
    arrow_props = dict(arrowstyle="->", linewidth=2, color="gray")

    # Input to Encoder
    ax.annotate("", xy=(0.05, 0.5), xytext=(-0.05, 0.5), arrowprops=arrow_props)
    ax.text(-0.1, 0.5, "Input Sequence", ha="right", va="center", fontsize=10)

    # Encoder to Attention
    ax.annotate("", xy=(0.45, 0.5), xytext=(0.35, 0.5), arrowprops=arrow_props)

    # Attention to Decoder
    ax.annotate("", xy=(0.65, 0.5), xytext=(0.55, 0.5), arrowprops=arrow_props)

    # Decoder to Output
    ax.annotate("", xy=(0.9, 0.5), xytext=(0.8, 0.5), arrowprops=arrow_props)
    ax.text(0.95, 0.5, "Output Sequence", ha="left", va="center", fontsize=10)

    # Add title
    ax.set_title("Sequence-to-Sequence Model Architecture with Attention", fontsize=14)

    # Add annotations
    textstr = "\n".join(
        [
            "Architecture Details:",
            "• Encoder: Bidirectional GRU encodes source sequence into context",
            "• Attention: Focuses on relevant parts of encoded sequence during decoding",
            "• Decoder: GRU with attention generates target sequence",
            "• Teacher forcing: During training, ground truth is fed as input with probability 0.5",
        ]
    )

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.4)
    ax.text(
        0.5,
        0.15,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        va="center",
        ha="center",
        bbox=props,
    )

    # Save and show
    plt.savefig("models/architecture.png", bbox_inches="tight", dpi=300)
    print("Architecture visualization saved to: models/architecture.png")
    plt.close()


def main():
    """Main function to create visualizations."""
    # Ensure model directory exists
    os.makedirs("models", exist_ok=True)

    # Create a combined visualization of model comparisons
    visualize_model_comparison(save_dir="models", show_plot=False)

    # Create architecture visualization
    visualize_architecture()

    print(
        "All visualizations complete. Check the models directory for the output files."
    )


if __name__ == "__main__":
    main()

