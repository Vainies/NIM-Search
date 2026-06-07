# NIM Search

Free AI search using NVIDIA NIM API. No paid search APIs needed.

Query multiple free AI models in parallel, then cross-reference and summarize the results. Works with OpenClaw, Hermes, or any tool that needs web search.

## Setup

1. Get a free API key at https://build.nvidia.com (no credit card)
2. Set it:
```bash
export NVIDIA_API_KEY="your-key-here"
```

## Usage

```bash
# Basic search
python3 nimsearch.py "what is linux namespaces"

# JSON output (for integration)
python3 nimsearch.py "best python web frameworks" --json

# Skip summarization (faster, raw model outputs only)
python3 nimsearch.py "query" --pass1-only

# Pick specific models
python3 nimsearch.py "query" --models maverick,kimi-k2.6,deepseek-v4-flash

# List all available models
python3 nimsearch.py --list-models
```

## Default Models

| Shortname | Model | Why |
|-----------|-------|-----|
| maverick | Llama 4 Maverick | Most popular, 22M uses, well-rounded |
| nemotron-70b | Nemotron 70B | Best for search and reasoning |
| minimax-m2.7 | MiniMax M2.7 | 230B MoE, competes with Claude |

## All Available Models (30+)

**Top picks for search:**
- `maverick` - Llama 4 Maverick (Meta, 17B MoE)
- `nemotron-70b` - Nemotron 70B (NVIDIA)
- `minimax-m2.7` - MiniMax M2.7 (230B MoE)
- `kimi-k2.6` - Kimi K2.6 (Moonshot AI, great for search)
- `mistral-large-3` - Mistral Large 3 (675B)
- `deepseek-v4-flash` - DeepSeek V4 Flash (fast reasoning)

**Also available:**
- `llama-3.3` - Llama 3.3 70B (Meta)
- `mistral-nemotron` - Mistral Nemotron (agentic, function calling)
- `gemma-4-31b` - Gemma 4 31B (Google)
- `phi-4` - Phi-4 Mini (Microsoft, fast)
- `seed-oss-36b` - Seed OSS 36B (ByteDance)
- `yi-large` - Yi Large (01.AI, multilingual)
- `jamba-1.5` - Jamba 1.5 Large (AI21, long context)
- And more - run `--list-models` for the full list

## How It Works

1. **Pass 1** - Sends your query to 3+ models in parallel, collects raw answers
2. **Pass 2** - A summarizer model cross-references all answers, extracts links, formats a clean response
3. Rate limiter stays under 38 RPM (free tier is 40)

## Rate Limits

- 40 requests per minute (free tier, no credit card)
- Tool stays under 38 RPM to be safe
- Auto-waits if approaching limit

## Integration with OpenClaw

Set `NVIDIA_API_KEY` in your environment. The `--json` flag outputs structured data that other tools can parse.

## Requirements

- Python 3.8+
- No external dependencies (stdlib only)
- NVIDIA NIM API key (free)

## License

MIT
