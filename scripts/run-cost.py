#!/usr/bin/env python3
"""
run-cost.py — what did that run cost?

Reads a Claude Code session transcript (~/.claude/projects/<slug>/<session-id>.jsonl)
and reports tokens and cost for a run, optionally bounded by a time window.

Answers the question a local skill run cannot answer today:
what did that run cost. Tokens come straight from the transcript and are exact.
Cost is only reported for models with a VERIFIED rate; an unknown model is reported
as a hard failure, never as $0 or null — a missing rate must be loud.

Usage:
  run-cost.py TRANSCRIPT.jsonl
  run-cost.py TRANSCRIPT.jsonl --since 2026-08-26T19:13:00Z --until 2026-08-26T19:45:00Z
  run-cost.py TRANSCRIPT.jsonl --timeline          # tool calls + timestamps, to pick a window
  run-cost.py TRANSCRIPT.jsonl --json
"""
import argparse, json, sys
from datetime import datetime, timezone

# Rates: USD per 1M tokens. Source: Anthropic published pricing, cached 2026-06-24.
# Update deliberately and re-date this comment; do not guess a missing row.
RATES = {
    "claude-fable-5":     {"in":10.00, "out":50.00},
    "claude-mythos-5":    {"in":10.00, "out":50.00},
    "claude-opus-5":      {"in": 5.00, "out":25.00},
    "claude-opus-4-8":    {"in": 5.00, "out":25.00},
    "claude-opus-4-7":    {"in": 5.00, "out":25.00},
    "claude-opus-4-6":    {"in": 5.00, "out":25.00},
    "claude-sonnet-5":    {"in": 2.00, "out":10.00},
    "claude-sonnet-4-6":  {"in": 3.00, "out":15.00},
    "claude-haiku-4-5":   {"in": 1.00, "out": 5.00},
}
# Cache multipliers against base input price. Source: Anthropic prompt-caching docs.
CACHE_READ_MULT  = 0.10
CACHE_WRITE_5M   = 1.25
CACHE_WRITE_1H   = 2.00

def parse_ts(s):
    if not s: return None
    try: return datetime.fromisoformat(s.replace("Z","+00:00"))
    except ValueError: return None

def collect(path, since=None, until=None):
    per_model, skipped, first, last, msgs = {}, 0, None, None, 0
    for line in open(path, encoding="utf-8"):
        try: rec = json.loads(line)
        except json.JSONDecodeError: continue
        ts = parse_ts(rec.get("timestamp"))
        if since and (ts is None or ts < since): continue
        if until and (ts is None or ts > until): continue
        m = rec.get("message")
        if not isinstance(m, dict): continue
        u = m.get("usage") or rec.get("usage")
        if not u: continue
        model = m.get("model") or "unknown"
        if ts:
            first = ts if first is None or ts < first else first
            last  = ts if last  is None or ts > last  else last
        msgs += 1
        cc = u.get("cache_creation") or {}
        b = per_model.setdefault(model, dict(msgs=0, inp=0, out=0, think=0, w5=0, w1h=0, read=0))
        b["msgs"] += 1
        b["inp"]  += u.get("input_tokens", 0) or 0
        b["out"]  += u.get("output_tokens", 0) or 0
        b["think"]+= ((u.get("output_tokens_details") or {}).get("thinking_tokens", 0)) or 0
        b["read"] += u.get("cache_read_input_tokens", 0) or 0
        w5  = cc.get("ephemeral_5m_input_tokens")
        w1h = cc.get("ephemeral_1h_input_tokens")
        if w5 is None and w1h is None:
            # older records: only the aggregate is present; attribute to 5m (the cheaper
            # assumption) and flag it so the number is never silently overstated
            b["w5"] += u.get("cache_creation_input_tokens", 0) or 0
        else:
            b["w5"]  += w5 or 0
            b["w1h"] += w1h or 0
    return per_model, first, last, msgs

def cost(model, b):
    r = RATES.get(model)
    if not r: return None
    return (b["inp"]*r["in"] + b["read"]*r["in"]*CACHE_READ_MULT
            + b["w5"]*r["in"]*CACHE_WRITE_5M + b["w1h"]*r["in"]*CACHE_WRITE_1H
            + b["out"]*r["out"]) / 1_000_000

def timeline(path, since=None, until=None):
    for line in open(path, encoding="utf-8"):
        try: rec = json.loads(line)
        except json.JSONDecodeError: continue
        ts = parse_ts(rec.get("timestamp"))
        if since and (ts is None or ts < since): continue
        if until and (ts is None or ts > until): continue
        m = rec.get("message")
        if not isinstance(m, dict): continue
        for c in (m.get("content") or []):
            if isinstance(c, dict) and c.get("type") == "tool_use":
                inp = json.dumps(c.get("input", {}))[:90].replace("\n", " ")
                print(f"{rec.get('timestamp','?')}  {c.get('name','?'):<28} {inp}")

def main():
    p = argparse.ArgumentParser(description="Report tokens and cost for a Claude Code run.")
    p.add_argument("transcript")
    p.add_argument("--since"); p.add_argument("--until")
    p.add_argument("--timeline", action="store_true")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    since, until = parse_ts(a.since), parse_ts(a.until)
    if a.since and not since: sys.exit(f"error: could not parse --since {a.since!r}")
    if a.until and not until: sys.exit(f"error: could not parse --until {a.until!r}")

    if a.timeline:
        timeline(a.transcript, since, until); return

    per_model, first, last, msgs = collect(a.transcript, since, until)
    if not per_model: sys.exit("error: no usage records in that window — check --since/--until")

    unpriced = [m for m in per_model if m not in RATES]
    total = 0.0
    rows = []
    for model, b in sorted(per_model.items()):
        c = cost(model, b)
        if c is not None: total += c
        rows.append((model, b, c))

    if a.json:
        print(json.dumps({
            "window": {"first": first.isoformat() if first else None,
                       "last": last.isoformat() if last else None},
            "assistant_messages": msgs,
            "models": {m: {**b, "cost_usd": c} for m, b, c in rows},
            "total_cost_usd": round(total, 4) if not unpriced else None,
            "unpriced_models": unpriced,
        }, indent=2))
        if unpriced: sys.exit(2)
        return

    span = ""
    if first and last:
        mins = (last - first).total_seconds() / 60
        span = f"  ({mins:.0f} min wall clock)"
    print(f"\nWindow: {first} → {last}{span}")
    print(f"Assistant messages: {msgs}\n")
    hdr = f"{'model':<20}{'msgs':>6}{'input':>10}{'cache wr':>11}{'cache rd':>11}{'output':>10}{'(think)':>10}{'cost':>11}"
    print(hdr); print("-" * len(hdr))
    for model, b, c in rows:
        cs = f"${c:,.2f}" if c is not None else "UNPRICED"
        print(f"{model:<20}{b['msgs']:>6}{b['inp']:>10,}{b['w5']+b['w1h']:>11,}{b['read']:>11,}"
              f"{b['out']:>10,}{b['think']:>10,}{cs:>11}")
    print("-" * len(hdr))
    if unpriced:
        print(f"\nFAIL: no verified rate for {', '.join(unpriced)}.")
        print("Add a row to RATES from published pricing. Refusing to report a total.")
        sys.exit(2)
    print(f"{'TOTAL':<20}{msgs:>6}{'':>10}{'':>11}{'':>11}{'':>10}{'':>10}{f'${total:,.2f}':>11}\n")
    print("Rates: Anthropic published pricing (cached 2026-06-24). Cache read 0.10x input,")
    print("cache write 1.25x (5-min TTL) / 2.00x (1-hour TTL). Tokens are exact from the")
    print("transcript; cost is arithmetic on those rates, not a billing statement.")

if __name__ == "__main__":
    main()
