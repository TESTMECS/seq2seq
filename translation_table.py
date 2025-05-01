#!/usr/bin/env python
"""
Script to display example English sentences and their French translations side by side in a chart.
Example DATA.
"""

import matplotlib.pyplot as plt
import argparse


def create_translation_chart(
    sources,
    translations,
    save_path=None,
    show_plot=True,
    max_length=None,
    title="English to French Translation Results",
):
    """Create a chart displaying source sentences and their translations side by side."""
    # Prepare data
    if max_length:
        sources = [
            s[:max_length] + "..." if len(s) > max_length else s for s in sources
        ]
        translations = [
            t[:max_length] + "..." if len(t) > max_length else t for t in translations
        ]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, len(sources) * 0.5 + 1.5))

    # Hide axes
    ax.axis("off")

    # Create table
    table_data = [[src, trans] for src, trans in zip(sources, translations)]

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=["English Source", "French Translation"],
        loc="center",
        cellLoc="left",
        colWidths=[0.5, 0.5],
        colColours=["#e6f2ff", "#ffe6e6"],
    )

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Adjust cell height
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Header row
            cell.set_height(0.1)
            cell.set_text_props(weight="bold")
        else:  # Data rows
            cell.set_height(0.08)
            # Alternate row colors
            if key[0] % 2 == 1:
                cell.set_facecolor("#f5f5f5")

    # Add title
    plt.suptitle(title, fontsize=14)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Chart saved to {save_path}")

    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """Main function to parse arguments and display translation chart."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Display English to French translations in a chart"
    )

    # Display parameters
    parser.add_argument(
        "--save_path",
        type=str,
        default="models/translation_chart.png",
        help="Path to save the visualization",
    )
    parser.add_argument(
        "--show_plot",
        action="store_true",
        default=True,
        help="Whether to display the plot",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum length of sentences to display",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="English to French Translation Results",
        help="Title for the chart",
    )
    parser.add_argument(
        "--english", type=str, nargs="+", help="Custom English sentences to translate"
    )
    parser.add_argument(
        "--french",
        type=str,
        nargs="+",
        help="Custom French translations (must match number of English sentences)",
    )

    args = parser.parse_args()

    # Default example sentences
    english_sentences = [
        "Hello, how are you?",
        "I love programming in Python.",
        "The weather is nice today.",
        "What time is it?",
        "Can you help me with this translation?",
        "I'm learning machine translation.",
        "This is a sequence-to-sequence model.",
        "Neural networks are powerful for language tasks.",
        "Attention mechanisms improve translation quality.",
        "Thank you for your help!",
    ]

    # Default example translations
    french_translations = [
        "Bonjour, comment allez-vous ?",
        "J'adore programmer en Python.",
        "Le temps est agréable aujourd'hui.",
        "Quelle heure est-il ?",
        "Pouvez-vous m'aider avec cette traduction ?",
        "J'apprends la traduction automatique.",
        "C'est un modèle séquence à séquence.",
        "Les réseaux de neurones sont puissants pour les tâches linguistiques.",
        "Les mécanismes d'attention améliorent la qualité de la traduction.",
        "Merci pour votre aide !",
    ]

    # Use custom sentences if provided
    if args.english and args.french:
        if len(args.english) != len(args.french):
            print("Error: Number of English and French sentences must match.")
            return
        english_sentences = args.english
        french_translations = args.french
    elif args.english or args.french:
        print(
            "Warning: Both English and French sentences must be provided to use custom input."
        )
        print("Using default examples instead.")

    # Display translations
    print("Creating translation chart with example data...")
    create_translation_chart(
        sources=english_sentences,
        translations=french_translations,
        save_path=args.save_path,
        show_plot=args.show_plot,
        max_length=args.max_length,
        title=args.title,
    )


if __name__ == "__main__":
    main()
