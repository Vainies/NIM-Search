---
name: nim-search
description: Free AI search using NVIDIA NIM API. Replaces paid search APIs (Brave, SerpAPI, etc.) with 30+ free models.
metadata:
  author: Vainies
  version: "1.0"
  category: search
---

# NIM Search Skill

Use this skill when you need to search the web or answer factual queries without paying for search APIs. Uses NVIDIA NIM's free tier (40 RPM, no credit card).

## When to Use

- User asks a question that requires current/web information
- You need to search the web but don't have Brave/SerpAPI access
- You want to cross-reference multiple AI model answers
- Any task that would normally use a paid search API

## How It Works

1. **Pass 1**: Query 3+ NVIDIA NIM models in parallel (Maverick, Nemotron 70B, MiniMax M2.7)
2. **Pass 2**: A summarizer model cross-references answers, extracts links, formats response
3. Output: Clean summary with sources

## Setup (One-Time)

```bash
# 1. Get free API key at https://build.nvidia.com (no credit card)
# 2. Clone the tool
git clone https://github.com/Vainies/NIM-Search.git /tmp/nim-search
# 3. Set the API key
export NVIDIA_API_KEY="***"
```

## Usage

```bash
# Basic search
python3 /tmp/nim-search/nimsearch.py "what is linux namespaces"

# JSON output (for piping to other tools)
python3 /tmp/nim-search/nimsearch.py "query" --json

# Fast mode (skip summarization)
python3 /tmp/nim-search/nimsearch.py "query" --pass1-only

# Pick specific models
python3 /tmp/nim-search/nimsearch.py "query" --models maverick,kimi-k2.6

# List all 30+ available models
python3 /tmp/nim-search/nimsearch.py --list-models
```

## Default Models

| Shortname | Model | Notes |
|-----------|-------|-------|
| maverick | Llama 4 Maverick | Most popular, 22M uses |
| nemotron-70b | Nemotron 70B | Best for search/reasoning |
| minimax-m2.7 | MiniMax M2.7 | 230B MoE, competes with Claude |

## Rate Limits

- Free tier: 40 requests per minute
- Tool stays under 38 RPM automatically
- Auto-waits if approaching limit

## Integration

When using this skill, parse the `--json` output:

```python
import subprocess, json
result = subprocess.run(
    ["python3", "/tmp/nim-search/nimsearch.py", query, "--json"],
    capture_output=True, text=True
)
data = json.loads(result.stdout)
# data["summary"] = clean answer
# data["sources"] = raw model responses
```

## Notes

- Requires Python 3.8+ (no external dependencies)
- NVIDIA API key is free, no credit card needed
- 30+ models available including DeepSeek, Mistral, Kimi, Gemma, etc.
