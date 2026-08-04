"""Offline evaluation for the wikivoyage RAG pipeline.

Two metrics:
1. Retrieval hit rate — does an expected city appear in top-k files?
2. Answer faithfulness — an LLM judge scores whether the answer is
   supported by the retrieved excerpts (1-5), run only when a Groq key
   is set (fast + cheap).

Results are appended to eval/history/ and compared against the previous
run so a regression is visible at a glance.

Run:
    python -m eval.run_eval [--top-k 5] [--no-judge]
"""

import argparse
import json
import time
from pathlib import Path

from src.config import GROQ_API_KEY, utf8_stdout
from src.rag import _stream_groq, build_context
from src.retrieval import hybrid_search

utf8_stdout()

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN = EVAL_DIR / "golden.jsonl"
HISTORY = EVAL_DIR / "history"

_JUDGE_SYSTEM = (
    "You grade whether an ANSWER is supported by the provided EXCERPTS. "
    "Reply with only a JSON object: {\"score\": 1-5, \"reason\": \"<short>\"}. "
    "5 = every claim is backed by the excerpts; 1 = mostly unsupported or "
    "contradicted. Ignore style; judge grounding only."
)


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def retrieval_hit(files: list[str], expected: list[str]) -> bool:
    """An expected city name matching any retrieved file (prefix, case-
    insensitive) counts as a hit."""
    low = [f.lower() for f in files]
    return any(any(f.startswith(e.lower()) for f in low) for e in
               [x.lower() for x in expected])


def judge_faithfulness(question: str, answer: str, context: str) -> dict:
    prompt = (f"EXCERPTS:\n{context}\n\nQUESTION: {question}\n\nANSWER:\n{answer}")
    try:
        raw = "".join(_stream_groq(None, prompt, None, _JUDGE_SYSTEM)).strip()
        raw = raw[raw.find("{"): raw.rfind("}") + 1]
        data = json.loads(raw)
        return {"score": int(data["score"]), "reason": data.get("reason", "")}
    except Exception as e:
        return {"score": 0, "reason": f"judge failed: {e}"}


def run(top_k: int, judge: bool) -> dict:
    golden = load_golden()
    hits_ok = 0
    faithfulness: list[int] = []
    rows = []

    for item in golden:
        hits = hybrid_search(item["q"], top_k=top_k, collection="wikivoyage")
        files = [h.file for h in hits]
        hit = retrieval_hit(files, item["expect"])
        hits_ok += hit
        row = {"q": item["q"], "expect": item["expect"], "files": files, "hit": hit}

        # Faithfulness is only meaningful when retrieval succeeded.
        if judge and GROQ_API_KEY and hits and hit:
            ctx = build_context(hits)
            try:
                answer = "".join(_stream_groq(ctx, item["q"]))
                verdict = judge_faithfulness(item["q"], answer, ctx)
                row["faithfulness"] = verdict["score"]
                if verdict["score"]:
                    faithfulness.append(verdict["score"])
            except RuntimeError as e:
                row["faithfulness"] = None
                row["judge_error"] = str(e)
            # Free-tier tokens/min is the binding limit; pace the calls.
            time.sleep(2.5)
        rows.append(row)

    n = len(golden)
    summary = {
        "n": n,
        "retrieval_hit_rate": round(hits_ok / n, 3),
        "avg_faithfulness": round(sum(faithfulness) / len(faithfulness), 2)
        if faithfulness else None,
        "top_k": top_k,
    }
    return {"summary": summary, "rows": rows}


def previous_summary() -> dict | None:
    if not HISTORY.exists():
        return None
    files = sorted(HISTORY.glob("eval-*.json"), reverse=True)
    if not files:
        return None
    return json.loads(files[0].read_text(encoding="utf-8"))["summary"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--stamp", default=None,
                        help="timestamp label for the history file")
    args = parser.parse_args()

    prev = previous_summary()
    result = run(args.top_k, judge=not args.no_judge)
    s = result["summary"]

    print("== Evaluation ==")
    print(f"  questions:          {s['n']}")
    print(f"  retrieval hit rate: {s['retrieval_hit_rate']:.1%}")
    if s["avg_faithfulness"] is not None:
        print(f"  avg faithfulness:   {s['avg_faithfulness']}/5")
    if prev:
        d = s["retrieval_hit_rate"] - prev["retrieval_hit_rate"]
        arrow = "▲" if d > 0 else ("▼" if d < 0 else "=")
        print(f"  vs previous:        {arrow} {d:+.1%} hit rate "
              f"(was {prev['retrieval_hit_rate']:.1%})")

    misses = [r["q"] for r in result["rows"] if not r["hit"]]
    if misses:
        print(f"  retrieval misses:   {len(misses)}")
        for q in misses:
            print(f"    - {q}")

    HISTORY.mkdir(exist_ok=True)
    stamp = args.stamp or "latest"
    (HISTORY / f"eval-{stamp}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
