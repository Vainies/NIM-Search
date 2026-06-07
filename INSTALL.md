# NIM Search - Self-Install for AI Agents

Quick setup guide for AI assistants to install and use NIM Search.

## Auto-Install

Run this to install NIM Search:

```bash
cd /tmp && git clone https://github.com/Vainies/NIM-Search.git 2>/dev/null || true
export NVIDIA_API_KEY="***"  # User must provide this
```

## Check If Installed

```bash
test -f /tmp/NIM-Search/nimsearch.py && echo "INSTALLED" || echo "NOT INSTALLED"
```

## Usage Pattern

When the user asks a question requiring web search:

1. Check if NIM Search is installed
2. If not, clone the repo (skip if already exists)
3. Check if NVIDIA_API_KEY is set
4. If not, ask the user to get one free at https://build.nvidia.com
5. Run the search:
```bash
python3 /tmp/NIM-Search/nimsearch.py "USER_QUERY" --json
```
6. Parse the JSON response and present the summary to the user

## Full Auto-Install Script

```bash
#!/bin/bash
# Auto-install NIM Search for AI agents

REPO_DIR="/tmp/NIM-Search"

# Clone if not exists
if [ ! -f "$REPO_DIR/nimsearch.py" ]; then
    echo "Installing NIM Search..."
    cd /tmp && git clone https://github.com/Vainies/NIM-Search.git
    echo "Installed to $REPO_DIR"
fi

# Check API key
if [ -z "$NVIDIA_API_KEY" ]; then
    echo "ERROR: NVIDIA_API_KEY not set"
    echo "Get a free key at: https://build.nvidia.com"
    echo "Then run: export NVIDIA_API_KEY=\"your_key_here\""
    exit 1
fi

echo "NIM Search ready"
echo "Usage: python3 $REPO_DIR/nimsearch.py \"your query\" --json"
```

## Model Shortcuts

For quick searches, use these model combinations:

```bash
# Fast (2 models, ~5s)
python3 nimsearch.py "query" --models maverick,nemotron-70b --pass1-only

# Balanced (3 models, ~10s)
python3 nimsearch.py "query" --models maverick,nemotron-70b,minimax-m2.7

# Deep (5 models, ~15s)
python3 nimsearch.py "query" --models maverick,nemotron-70b,minimax-m2.7,kimi-k2.6,deepseek-v4-flash

# Code-focused
python3 nimsearch.py "query" --models minimax-m2.7,codestral

# Multilingual
python3 nimsearch.py "query" --models yi-large,kimi-k2.6
```

## Integration Example

```python
import subprocess
import json
import os

def nim_search(query, models=None):
    """Search using NIM Search and return summary."""
    cmd = ["python3", "/tmp/NIM-Search/nimsearch.py", query, "--json"]
    if models:
        cmd.extend(["--models", models])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"error": result.stderr}

    return json.loads(result.stdout)

# Example usage
result = nim_search("what are linux namespaces")
print(result["summary"])
```

## Troubleshooting

- **"No API key"** -> Set NVIDIA_API_KEY env var
- **Rate limit hit** -> Tool auto-waits, or use fewer models
- **Model not found** -> Run `--list-models` to see available models
