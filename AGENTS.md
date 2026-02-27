# Agent Instructions — fast-associative-memory

## Token-Dense Files: DO NOT READ

The following paths contain large data, binary, or generated content. Never read these files in full. Use grep or chatsearch to find specific information instead.

### Data directories (binary/archive/large text)
- `data/` — contains tar.gz, tar, tgz archives and large text files (e.g., `image_attribute_labels.txt` at 3.6M lines)
- `results/**/*.json` — experiment result JSON files (e.g., `results.json` at 103K lines)
- `results/**/*.csv` — experiment result CSVs (e.g., `g36_per_query_influence.csv` at 50K lines, `mt5_paths.csv` at 12K lines)

### Virtual environment
- `.venv/` — Python virtual environment. Never read files inside `.venv/`. Use `pip list` or `pip show <package>` if you need dependency info.

## Preferred Discovery Method

1. Use `chatsearch_find` or `chatsearch_ask` to locate relevant code and context.
2. For experiment results, read only the first 20 lines to understand the schema, then use grep for specific values.
3. For source code, use grep to find relevant functions/classes before reading files.
