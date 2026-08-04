"""Prefect flow wrapping the offline evaluation.

Schedule this (or run on every model/prompt change) so retrieval hit rate
and answer faithfulness are tracked over time, not eyeballed once.

    python -m flows.eval_flow
"""

from prefect import flow, task

from eval.run_eval import previous_summary, run


@task
def evaluate(top_k: int, judge: bool) -> dict:
    return run(top_k=top_k, judge=judge)


@task
def check_regression(summary: dict, prev: dict | None) -> None:
    if prev is None:
        print("No previous run to compare against.")
        return
    delta = summary["retrieval_hit_rate"] - prev["retrieval_hit_rate"]
    if delta < -0.05:
        print(f"REGRESSION: hit rate dropped {delta:.1%} "
              f"({prev['retrieval_hit_rate']:.1%} -> {summary['retrieval_hit_rate']:.1%})")
    else:
        print(f"OK: hit rate {summary['retrieval_hit_rate']:.1%} "
              f"(delta {delta:+.1%})")


@flow(name="rag-eval")
def eval_flow(top_k: int = 5, judge: bool = False):
    prev = previous_summary()
    result = evaluate(top_k, judge)
    check_regression(result["summary"], prev)
    return result["summary"]


if __name__ == "__main__":
    print(eval_flow())
