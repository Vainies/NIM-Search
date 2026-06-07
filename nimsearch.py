#!/usr/bin/env python3
"""
NIM Search - Free AI search using NVIDIA NIM API.

Two-pass system:
  Pass 1: Query multiple free AI models in parallel for raw answers
  Pass 2: Summarize, cross-reference, and extract links

Replaces paid search APIs (Brave, etc.) with free NVIDIA NIM models.

Usage:
    python3 nimsearch.py "what is linux namespaces"
    python3 nimsearch.py "best python web frameworks" --json
    python3 nimsearch.py "query" --pass1-only
    python3 nimsearch.py --list-models

Requirements: Python 3.8+, no external dependencies
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


API_BASE = "https://integrate.api.nvidia.com/v1"


# ── All free models on NVIDIA NIM (June 2026) ──
# Filtered to chat/instruct models only (excludes embeddings, classifiers, etc.)
ALL_MODELS = {
    # NVIDIA
    "nemotron-70b": {
        "id": "nvidia/llama-3.1-nemotron-70b-instruct",
        "name": "Nemotron 70B",
        "publisher": "NVIDIA",
        "size": "70B",
        "best_for": "search, reasoning, instruction following",
    },
    "nemotron-51b": {
        "id": "nvidia/llama-3.1-nemotron-51b-instruct",
        "name": "Nemotron 51B",
        "publisher": "NVIDIA",
        "size": "51B",
        "best_for": "balanced performance",
    },
    "nemotron-nano": {
        "id": "nvidia/llama-3.1-nemotron-nano-8b-v1",
        "name": "Nemotron Nano 8B",
        "publisher": "NVIDIA",
        "size": "8B",
        "best_for": "fast responses, low latency",
    },

    # Meta
    "maverick": {
        "id": "meta/llama-4-maverick-17b-128e-instruct",
        "name": "Llama 4 Maverick",
        "publisher": "Meta",
        "size": "17B MoE",
        "best_for": "general purpose, most popular (22M uses)",
    },
    "llama-3.3": {
        "id": "meta/llama-3.3-70b-instruct",
        "name": "Llama 3.3 70B",
        "publisher": "Meta",
        "size": "70B",
        "best_for": "reasoning, complex tasks",
    },
    "llama-3.1-70b": {
        "id": "meta/llama-3.1-70b-instruct",
        "name": "Llama 3.1 70B",
        "publisher": "Meta",
        "size": "70B",
        "best_for": "general purpose, well-tested",
    },
    "llama-3.1-8b": {
        "id": "meta/llama-3.1-8b-instruct",
        "name": "Llama 3.1 8B",
        "publisher": "Meta",
        "size": "8B",
        "best_for": "fast, lightweight",
    },
    "llama-3.2-90b-vision": {
        "id": "meta/llama-3.2-90b-vision-instruct",
        "name": "Llama 3.2 90B Vision",
        "publisher": "Meta",
        "size": "90B",
        "best_for": "multimodal (image + text)",
    },

    # Mistral
    "mistral-large-3": {
        "id": "mistralai/mistral-large-3-675b-instruct-2512",
        "name": "Mistral Large 3 (675B)",
        "publisher": "Mistral",
        "size": "675B",
        "best_for": "top-tier reasoning, professional",
    },
    "mistral-nemotron": {
        "id": "mistralai/mistral-nemotron",
        "name": "Mistral Nemotron",
        "publisher": "Mistral/NVIDIA",
        "size": "-",
        "best_for": "agentic workflows, function calling",
    },
    "mistral-medium": {
        "id": "mistralai/mistral-medium-3.5-128b",
        "name": "Mistral Medium 3.5",
        "publisher": "Mistral",
        "size": "128B",
        "best_for": "balanced, good value",
    },
    "mistral-small-4": {
        "id": "mistralai/mistral-small-4-119b-2603",
        "name": "Mistral Small 4",
        "publisher": "Mistral",
        "size": "119B",
        "best_for": "fast, efficient",
    },
    "codestral": {
        "id": "mistralai/codestral-22b-instruct-v0.1",
        "name": "Codestral 22B",
        "publisher": "Mistral",
        "size": "22B",
        "best_for": "code generation",
    },
    "ministral-14b": {
        "id": "mistralai/ministral-14b-instruct-2512",
        "name": "Ministral 14B",
        "publisher": "Mistral",
        "size": "14B",
        "best_for": "fast coding",
    },

    # MiniMax
    "minimax-m2.7": {
        "id": "minimaxai/minimax-m2.7",
        "name": "MiniMax M2.7",
        "publisher": "MiniMax",
        "size": "230B",
        "best_for": "coding, competes with Claude",
    },

    # Moonshot (Kimi)
    "kimi-k2.6": {
        "id": "moonshotai/kimi-k2.6",
        "name": "Kimi K2.6",
        "publisher": "Moonshot AI",
        "size": "-",
        "best_for": "search, long context, Chinese+English",
    },

    # DeepSeek
    "deepseek-v4-flash": {
        "id": "deepseek-ai/deepseek-v4-flash",
        "name": "DeepSeek V4 Flash",
        "publisher": "DeepSeek",
        "size": "-",
        "best_for": "fast reasoning, good at math",
    },
    "deepseek-v4-pro": {
        "id": "deepseek-ai/deepseek-v4-pro",
        "name": "DeepSeek V4 Pro",
        "publisher": "DeepSeek",
        "size": "-",
        "best_for": "complex reasoning, analysis",
    },

    # Google
    "gemma-4-31b": {
        "id": "google/gemma-4-31b-it",
        "name": "Gemma 4 31B",
        "publisher": "Google",
        "size": "31B",
        "best_for": "general purpose, well-rounded",
    },
    "gemma-3-12b": {
        "id": "google/gemma-3-12b-it",
        "name": "Gemma 3 12B",
        "publisher": "Google",
        "size": "12B",
        "best_for": "fast, lightweight",
    },

    # Microsoft
    "phi-4": {
        "id": "microsoft/phi-4-mini-instruct",
        "name": "Phi-4 Mini",
        "publisher": "Microsoft",
        "size": "-",
        "best_for": "fast, efficient, edge use",
    },
    "phi-3.5-moe": {
        "id": "microsoft/phi-3.5-moe-instruct",
        "name": "Phi-3.5 MoE",
        "publisher": "Microsoft",
        "size": "-",
        "best_for": "mixture of experts, balanced",
    },

    # ByteDance
    "seed-oss-36b": {
        "id": "bytedance/seed-oss-36b-instruct",
        "name": "Seed OSS 36B",
        "publisher": "ByteDance",
        "size": "36B",
        "best_for": "general purpose",
    },

    # IBM
    "granite-34b": {
        "id": "ibm/granite-34b-code-instruct",
        "name": "Granite 34B Code",
        "publisher": "IBM",
        "size": "34B",
        "best_for": "enterprise, code",
    },

    # 01.AI
    "yi-large": {
        "id": "01-ai/yi-large",
        "name": "Yi Large",
        "publisher": "01.AI",
        "size": "-",
        "best_for": "multilingual, reasoning",
    },

    # AI21
    "jamba-1.5": {
        "id": "ai21labs/jamba-1.5-large-instruct",
        "name": "Jamba 1.5 Large",
        "publisher": "AI21 Labs",
        "size": "-",
        "best_for": "long context, hybrid architecture",
    },

    # Abacus
    "dracarys-70b": {
        "id": "abacusai/dracarys-llama-3.1-70b-instruct",
        "name": "Dracarys 70B",
        "publisher": "Abacus AI",
        "size": "70B",
        "best_for": "instruct following, safety",
    },
}

# Default search team: fast, diverse, high quality
DEFAULT_SEARCH_MODELS = ["maverick", "nemotron-70b", "minimax-m2.7"]
DEFAULT_SUMMARIZER = "nemotron-70b"


class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    RESET = "\033[0m"


def colored(text, color):
    return f"{color}{text}{C.RESET}"


class NIMClient:
    def __init__(self, api_key=***
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "No API key. Set NVIDIA_API_KEY env var or pass --api-key.\n"
                "Get one free at https://build.nvidia.com"
            )
        self.base_url = API_BASE
        self.request_count = 0
        self.window_start = time.time()

    def _check_rate_limit(self):
        now = time.time()
        if now - self.window_start > 60:
            self.request_count = 0
            self.window_start = now
        if self.request_count >= 38:
            wait = 60 - (now - self.window_start)
            if wait > 0:
                print(colored(f"  Rate limit: waiting {wait:.0f}s...", C.DIM))
                time.sleep(wait)
                self.request_count = 0
                self.window_start = time.time()

    def chat(self, model_id, messages, temperature=0.3, max_tokens=1024):
        self._check_rate_limit()
        url = f"{self.base_url}/chat/completions"
        payload = json.dumps({
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        req = Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Accept", "application/json")

        start = time.time()
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self.request_count += 1
                elapsed = time.time() - start
                content = data["choices"][0]["message"]["content"]
                tokens = data.get("usage", {})
                return {
                    "ok": True,
                    "content": content,
                    "model": model_id,
                    "elapsed": elapsed,
                    "tokens": tokens,
                }
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return {
                "ok": False,
                "error": f"HTTP {e.code}: {body[:200]}",
                "model": model_id,
            }
        except (URLError, TimeoutError) as e:
            return {"ok": False, "error": str(e), "model": model_id}


SEARCH_SYSTEM = """You are a helpful search assistant. Answer the user's query
with factual, accurate information. Include specific details, numbers, and
facts. If you reference something that has a URL, include the URL.
Be concise but thorough. Focus on giving the user what they asked for.
If you are unsure about something, say so rather than making it up."""

def pass1_search(client, query, model_keys):
    messages = [
        {"role": "system", "content": SEARCH_SYSTEM},
        {"role": "user", "content": query},
    ]
    results = []
    with ThreadPoolExecutor(max_workers=len(model_keys)) as executor:
        futures = {}
        for key in model_keys:
            m = ALL_MODELS[key]
            futures[executor.submit(client.chat, m["id"], messages)] = (key, m)

        for future in as_completed(futures):
            key, model = futures[future]
            result = future.result()
            result["model_name"] = model["name"]
            result["model_key"] = key
            if result["ok"]:
                print(colored(f"  + {model['name']} ({result['elapsed']:.1f}s)", C.GREEN))
            else:
                print(colored(f"  x {model['name']}: {result['error'][:60]}", C.RED))
            results.append(result)
    return results


SUMMARIZE_SYSTEM = """You are a search result summarizer. You receive raw answers
from multiple AI models about the same query. Your job:

1. Cross-reference the answers for accuracy
2. Pick the most useful and accurate information
3. Extract any URLs/links mentioned
4. Provide a clean, well-formatted summary
5. If there are disagreements between sources, note it

Format your response as:
- A clear, concise answer
- Bullet points for key facts
- Links at the bottom if any were referenced
- Keep it under 300 words unless the topic demands more"""

def pass2_summarize(client, query, raw_results, summarizer_key=None):
    if not raw_results:
        return {"ok": False, "error": "No results to summarize"}

    sources = []
    for r in raw_results:
        if r["ok"]:
            sources.append(
                f"--- Source: {r['model_name']} ---\n{r['content']}"
            )

    if not sources:
        return {"ok": False, "error": "All model queries failed"}

    context = "\n\n".join(sources)
    summary_prompt = f"""Original query: {query}

Answers from {len(sources)} different AI models:

{context}

Provide a clean, cross-referenced summary of the above answers."""

    key = summarizer_key or DEFAULT_SUMMARIZER
    model = ALL_MODELS[key]
    messages = [
        {"role": "system", "content": SUMMARIZE_SYSTEM},
        {"role": "user", "content": summary_prompt},
    ]

    result = client.chat(model["id"], messages, temperature=0.1, max_tokens=1500)
    result["model_name"] = model["name"]
    return result


def list_models():
    print(colored("\n  Available Models on NVIDIA NIM\n", C.CYAN))
    print(f"  {'Shortname':<20} {'Model':<35} {'Size':<10} {'Best For'}")
    print(f"  {'-'*20} {'-'*35} {'-'*10} {'-'*30}")
    for key, m in sorted(ALL_MODELS.items()):
        print(f"  {colored(key, C.GREEN):<30} {m['name']:<35} {m['size']:<10} {m['best_for']}")
    print()


def print_results(query, raw, summary, elapsed_total):
    print()
    print(colored("  NIM Search Results", C.CYAN))
    print()
    print(f"  Query: {query}")
    print(f"  Models: {len([r for r in raw if r['ok']])} responded")
    print(f"  Time: {elapsed_total:.1f}s")
    print()

    if summary and summary.get("ok"):
        print(colored("  -- Summary --", C.BOLD))
        for line in summary["content"].split("\n"):
            if line.strip():
                print(f"  {line}")
        print()
    elif raw:
        for r in raw:
            if r["ok"]:
                print(colored(f"  -- {r['model_name']} --", C.BOLD))
                for line in r["content"].split("\n"):
                    if line.strip():
                        print(f"  {line}")
                print()
                break

    print(colored("  -- Sources --", C.DIM))
    for r in raw:
        if r["ok"]:
            print(f"    {colored(r['model_name'], C.GREEN)} ({r['elapsed']:.1f}s)")
        else:
            print(f"    {colored(r['model_name'], C.RED)} (failed)")
    print()


def output_json(query, raw, summary, elapsed_total):
    data = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "elapsed": round(elapsed_total, 2),
        "summary": summary.get("content") if summary and summary.get("ok") else None,
        "summary_model": summary.get("model") if summary else None,
        "sources": [
            {
                "model": r.get("model_name", r["model"]),
                "ok": r["ok"],
                "content": r.get("content"),
                "error": r.get("error"),
                "elapsed": r.get("elapsed"),
            }
            for r in raw
        ],
    }
    print(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="NIM Search - free AI search via NVIDIA NIM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  nimsearch.py "what is linux namespaces"
  nimsearch.py "best python web frameworks" --json
  nimsearch.py "query" --pass1-only
  nimsearch.py --list-models
  nimsearch.py "query" --models maverick,kimi-k2.6,deepseek-v4-flash

env:
  NVIDIA_API_KEY   your NVIDIA NIM API key (free at build.nvidia.com)
        """,
    )
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--api-key", help="NVIDIA API key (or set NVIDIA_API_KEY)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--pass1-only", action="store_true", help="Skip summarization")
    parser.add_argument("--list-models", action="store_true", help="List all available models")
    parser.add_argument(
        "--models",
        help="Comma-separated model shortnames (e.g. maverick,kimi-k2.6,nemotron-70b)",
        default=None,
    )
    parser.add_argument("--summarizer", help="Model shortname for summarization", default=None)

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    if not args.query:
        parser.error("query is required (or use --list-models)")

    # Parse model selection
    if args.models:
        model_keys = [k.strip() for k in args.models.split(",")]
        for k in model_keys:
            if k not in ALL_MODELS:
                print(colored(f"  Unknown model: {k}", C.RED))
                print(f"  Use --list-models to see available models")
                sys.exit(1)
    else:
        model_keys = DEFAULT_SEARCH_MODELS

    summarizer_key = args.summarizer
    if summarizer_key and summarizer_key not in ALL_MODELS:
        print(colored(f"  Unknown summarizer model: {summarizer_key}", C.RED))
        sys.exit(1)

    try:
        client = NIMClient(api_key=args.api_key)
    except ValueError as e:
        print(colored(f"  {e}", C.RED))
        sys.exit(1)

    print(colored(f"\n  Searching: {args.query}", C.BOLD))
    print(f"  Models: {', '.join(model_keys)}")
    print()

    start = time.time()

    print(colored("  Pass 1: Querying models...", C.DIM))
    raw = pass1_search(client, args.query, model_keys)

    summary = None
    if not args.pass1_only:
        print(colored("  Pass 2: Summarizing...", C.DIM))
        summary = pass2_summarize(client, args.query, raw, summarizer_key)

    elapsed = time.time() - start

    if args.json:
        output_json(args.query, raw, summary, elapsed)
    else:
        print_results(args.query, raw, summary, elapsed)


if __name__ == "__main__":
    main()
