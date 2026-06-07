#!/usr/bin/env python3
"""
NIM Search - Free AI search using NVIDIA NIM API.

Two-pass system:
  Pass 1: Query multiple free AI models in parallel for raw answers
  Pass 2: Summarize, extract links, format cleanly

Replaces paid search APIs (Brave, etc.) with free NVIDIA NIM models.

Usage:
    python3 nimsearch.py "what is linux namespaces"
    python3 nimsearch.py "best python web frameworks" --json
    python3 nimsearch.py --config models.yaml
    python3 nimsearch.py "query" --pass1-only

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

DEFAULT_MODELS = [
    {
        "id": "nvidia/llama-3.1-nemotron-70b-instruct",
        "name": "Nemotron 70B",
        "role": "searcher",
    },
    {
        "id": "meta/llama-4-maverick-17b-128e-instruct",
        "name": "Llama 4 Maverick",
        "role": "searcher",
    },
    {
        "id": "nvidia/nemotron-3-8b-chat-steerlm-filter",
        "name": "Nemotron 8B",
        "role": "searcher",
    },
]

SUMMARIZER = {
    "id": "nvidia/llama-3.1-nemotron-70b-instruct",
    "name": "Nemotron 70B (summarizer)",
    "role": "summarizer",
}


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
    def __init__(self, api_key=None):
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
Be concise but thorough. Focus on giving the user what they asked for."""

def pass1_search(client, query, models):
    messages = [
        {"role": "system", "content": SEARCH_SYSTEM},
        {"role": "user", "content": query},
    ]
    results = []
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(client.chat, m["id"], messages): m
            for m in models
        }
        for future in as_completed(futures):
            model = futures[future]
            result = future.result()
            result["model_name"] = model["name"]
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

def pass2_summarize(client, query, raw_results, summarizer_model=None):
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

    model = summarizer_model or SUMMARIZER["id"]
    messages = [
        {"role": "system", "content": SUMMARIZE_SYSTEM},
        {"role": "user", "content": summary_prompt},
    ]

    result = client.chat(model, messages, temperature=0.1, max_tokens=1500)
    result["model_name"] = SUMMARIZER["name"]
    return result


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

env:
  NVIDIA_API_KEY   your NVIDIA NIM API key (free at build.nvidia.com)
        """,
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument("--api-key", help="NVIDIA API key (or set NVIDIA_API_KEY)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--pass1-only", action="store_true", help="Skip summarization")
    parser.add_argument(
        "--models",
        help="Comma-separated model shortnames (llama, maverick, nemotron)",
        default=None,
    )
    parser.add_argument("--summarizer", help="Model to use for summarization", default=None)

    args = parser.parse_args()

    model_map = {
        "nemotron": DEFAULT_MODELS[0],
        "llama": DEFAULT_MODELS[1],
        "maverick": DEFAULT_MODELS[1],
        "nemotron-8b": DEFAULT_MODELS[2],
    }
    if args.models:
        selected = []
        for name in args.models.split(","):
            name = name.strip().lower()
            if name in model_map:
                selected.append(model_map[name])
            else:
                selected.append({"id": name, "name": name, "role": "searcher"})
        models = selected
    else:
        models = DEFAULT_MODELS

    if not args.pass1_only:
        if args.summarizer:
            if args.summarizer in model_map:
                summarizer = model_map[args.summarizer]
            else:
                summarizer = {"id": args.summarizer, "name": args.summarizer}
        else:
            summarizer = SUMMARIZER

    try:
        client = NIMClient(api_key=args.api_key)
    except ValueError as e:
        print(colored(f"  {e}", C.RED))
        sys.exit(1)

    print(colored(f"\n  Searching: {args.query}\n", C.BOLD))

    start = time.time()

    print(colored("  Pass 1: Querying models...", C.DIM))
    raw = pass1_search(client, args.query, models)

    summary = None
    if not args.pass1_only:
        print(colored("  Pass 2: Summarizing...", C.DIM))
        summary = pass2_summarize(client, args.query, raw)

    elapsed = time.time() - start

    if args.json:
        output_json(args.query, raw, summary, elapsed)
    else:
        print_results(args.query, raw, summary, elapsed)


if __name__ == "__main__":
    main()
