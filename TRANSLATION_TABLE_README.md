# Translation Table Visualization

This tool creates a visualization showing English sentences and their French translations side by side in a nicely formatted table.

## Features

- Display English-French sentence pairs in a clean, tabular format
- Customize with your own sentences and translations
- Set custom chart title
- Save output as a high-quality PNG image
- Control maximum sentence length displayed

## Usage

Run the script with the UV environment:

```bash
uv run translation_table.py [options]
```

### Options

- `--title TEXT` - Set a custom title for the chart (default: "English to French Translation Results")
- `--save_path PATH` - Path to save the output image (default: "models/translation_chart.png")
- `--max_length INT` - Maximum character length to display for each sentence (truncates longer sentences)
- `--english TEXT [TEXT ...]` - Custom English sentences to display (must be used with --french)
- `--french TEXT [TEXT ...]` - Custom French translations (must match number of English sentences)
- `--show_plot` - Show the plot (may not work in non-interactive environments)

### Examples

Using default example sentences:
```bash
uv run translation_table.py
```

Using a custom title:
```bash
uv run translation_table.py --title "My Translation Examples"
```

Using custom sentences and translations:
```bash
uv run translation_table.py --english "Hello world" "How are you?" --french "Bonjour le monde" "Comment allez-vous?"
```

Limiting sentence length:
```bash
uv run translation_table.py --max_length 20
```

Custom output path:
```bash
uv run translation_table.py --save_path "output/my_translations.png"
```

## Related Files

- `translation_chart.py` - Advanced version that integrates with the seq2seq model (when available)
- `translation_table.py` - Standalone version that works without the model for visualization only

## Extending

You can modify this script to:
- Add more language pairs
- Integrate with different translation models
- Add highlighting for specific words
- Include additional metrics (like confidence scores)
- Create interactive versions with libraries like Plotly