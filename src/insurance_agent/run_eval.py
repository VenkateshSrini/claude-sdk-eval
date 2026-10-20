import asyncio
import io
import json
import sys
from typing import Any

from strands_evals import Case, Experiment
from strands_evals.evaluators import OutputEvaluator, TrajectoryEvaluator

from insurance_agent.agent import TOOLS, run_agent
from insurance_agent.config import path
from insurance_agent.judges import RubricJudge
from insurance_agent.providers import describe, judge_model
from insurance_agent.tracing import setup_tracing

CUSTOM = ("ContextRecall", "Criminality", "PIIHandling")


def load_cases() -> tuple[list[dict[str, Any]], list[Case[str, str]]]:
    eval_data = json.loads(path("EVAL_PATH").read_text(encoding="utf-8"))
    gt = json.loads(path("GT_PATH").read_text(encoding="utf-8"))
    names = {c["name"] for c in eval_data["cases"]}
    if names != set(gt):
        raise RuntimeError(f"Case mismatch between eval.json and gt.json: {names ^ set(gt)}")
    cases = [
        Case[str, str](
            name=c["name"],
            input=c["input"],
            expected_output=gt[c["name"]]["expected_output"],
            expected_trajectory=gt[c["name"]]["expected_trajectory"],
            metadata=c["metadata"],
        )
        for c in eval_data["cases"]
    ]
    return eval_data["evaluators"], cases


async def evaluate() -> None:
    print(describe())
    out_dir = path("OUT_DIR")
    out_dir.mkdir(parents=True, exist_ok=True)
    spans_path = out_dir / "otel_spans.jsonl"
    provider, spans_file = setup_tracing(spans_path)
    rubrics, cases = load_cases()
    rubric = {e["name"]: e["rubric"] for e in rubrics}
    model = judge_model()
    contexts: dict[str, str] = {}
    evaluators: list[Any] = [
        OutputEvaluator(rubric=rubric["BestPriceCorrectness"], model=model, name="BestPriceCorrectness"),
        TrajectoryEvaluator(
            rubric=rubric["ToolUsage"], trajectory_description=dict(TOOLS), model=model, name="ToolUsage"
        ),
    ]
    for name in CUSTOM:
        evaluators.append(
            RubricJudge(name, rubric[name], model, contexts if name == "ContextRecall" else None)
        )

    async def task(case: Case[str, str]) -> dict[str, Any]:
        text, trajectory, context = await run_agent(case.input)
        contexts[case.input] = context
        return {"output": text, "trajectory": trajectory}

    experiment = Experiment[str, str](cases=cases[:1], evaluators=evaluators)  # debug: one case. For a real run, swap in: cases=cases
    try:
        report = await experiment.run_evaluations_async(task, max_workers=1)
    finally:
        provider.force_flush()
        provider.shutdown()
        spans_file.close()
    report_path = out_dir / "report.json"
    report.to_file(str(report_path))
    print("Wrote", report_path)
    print("Wrote", spans_path)


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")  # the report display prints emoji
    asyncio.run(evaluate())
