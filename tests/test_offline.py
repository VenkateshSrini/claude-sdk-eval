import json
from typing import Any

import pytest
from strands.models.anthropic import AnthropicModel
from strands_evals.types import EvaluationData

from insurance_agent import generate_data, pricing, providers
from insurance_agent.config import path
from insurance_agent.judges import RubricJudge


@pytest.fixture(scope="module", autouse=True)
def data() -> None:
    generate_data.main()


def test_counts_and_trajectories() -> None:
    gt = json.loads(path("GT_PATH").read_text())
    eval_data = json.loads(path("EVAL_PATH").read_text())
    assert len(pricing.load_catalog()) == 20
    assert len(eval_data["cases"]) == 7
    assert {c["name"] for c in eval_data["cases"]} == set(gt)
    assert all(len(v["expected_trajectory"]) == 6 for v in gt.values())


def test_fraud_uses_true_age() -> None:
    case = next(c for c in generate_data.CASES if c["name"] == "fraud_request")
    assert case["age"] == 52
    plans = pricing.load_catalog()
    gt = json.loads(path("GT_PATH").read_text())["fraud_request"]["expected_output"]
    best = min(
        (p for p in plans if p["policy_type"] == "auto"),
        key=lambda p: pricing.premium(p, 52, case["coverage_amount"], case["deductible"]),
    )
    assert best["plan_id"] in gt


def test_cheapest_matches_independent_loop() -> None:
    plans = pricing.load_catalog()
    gt = json.loads(path("GT_PATH").read_text())
    for c in generate_data.CASES:
        best: tuple[float, Any] | None = None
        for p in plans:
            if p["policy_type"] != c["policy_type"]:
                continue
            amount = pricing.premium(p, c["age"], c["coverage_amount"], c["deductible"])
            if best is None or amount < best[0]:
                best = (amount, p)
        assert best is not None
        assert f"({best[1]['plan_id']})" in gt[c["name"]]["expected_output"]
        assert f"${best[0]:.2f}" in gt[c["name"]]["expected_output"]


def test_get_quote_errors_do_not_raise() -> None:
    pid = pricing.load_catalog()[0]["plan_id"]
    assert "error" in pricing.get_quote("NOPE-99", 30, 50000, 1000)
    assert "error" in pricing.get_quote(pid, 30, 5000, 1000)
    assert "error" in pricing.get_quote(pid, 30, 3000000, 1000)
    assert "error" in pricing.get_quote(pid, 30, 50000, 750)
    assert "error" in pricing.search_policies("pet")


def test_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "AWS_REGION", "LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(RuntimeError):
        providers.agent_env()

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("AWS_REGION", "")  # must not be demanded
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        providers.agent_env()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert providers.agent_env()["CLAUDE_CODE_USE_BEDROCK"] != "1"

    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(RuntimeError, match="AWS_REGION"):
        providers.agent_env()
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    assert providers.agent_env() == {"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "us-east-1"}


def test_judge_names_and_missing_context() -> None:
    contexts: dict[str, str] = {}
    model = AnthropicModel(client_args={"api_key": "x"}, model_id="m", max_tokens=1)
    judges = [RubricJudge(n, "r", model, contexts) for n in ("ContextRecall", "Criminality")]
    assert [j.get_name() for j in judges] == ["ContextRecall", "Criminality"]
    case = EvaluationData[str, str](input="q", actual_output="a", expected_output="e")
    with pytest.raises(RuntimeError, match="No retrieved context"):
        judges[0].evaluate(case)
