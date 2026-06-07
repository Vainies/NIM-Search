# NIM Search

Free AI search using NVIDIA NIM API. No paid search APIs needed.

Two-pass system:
1. Query multiple free AI models in parallel for raw answers
2. Summarize, cross-reference, and extract links

Works with OpenClaw, Hermes, or any tool that needs web search.

## Setup

1. Get a free API key at https://build.nvidia.com
2. Set it:
```bash
export NVIDIA_API_KEY="nvapi-xxxx"
```

## Usage

```bash
# Basic search
python3 nimsearch.py "what is linux namespaces"

# JSON output
python3 nimsearch.py "best python web frameworks" --json

# Skip summarization (faster)
python3 nimsearch.py "query" --pass1-only

# Pick specific models
python3 nimsearch.py "query" --models llama,maverick,nemotron
```

## Available Models (Free)

| Shortname | Model | Notes |
|-----------|-------|-------|
| nemotron | Nemotron 70B | Best for search/grounding |
| llama | Llama 4 Maverick | Most popular, 22M uses |
| maverick | Llama 4 Maverick | Same as llama |
| nemotron-8b | Nemotron 8B | Fastest, lighter |

## Integration with OpenClaw

Set NVIDIA_API_KEY in your environment and use this as a search provider.

## Rate Limits

- 40 requests per minute (free tier)
- Tool stays under 38 RPM to be safe
- Auto-waits if approaching limit

## Requirements

- Python 3.8+
- No external dependencies (uses stdlib only)
- NVIDIA NIM API key (free)

## License

MIT
