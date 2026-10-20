import json
from typing import Any

from insurance_agent.config import path

POLICY_TYPES = ("auto", "home", "term_life", "health")
DEDUCTIBLE_FACTORS = {500: 1.15, 1000: 1.0, 2500: 0.85, 5000: 0.72}

Plan = dict[str, Any]


def load_catalog() -> list[Plan]:
    plans: list[Plan] = json.loads(path("DATA_PATH").read_text(encoding="utf-8"))
    return plans


def age_factor(policy_type: str, age: int) -> float:
    if policy_type == "term_life":
        return 1 + max(0, age - 25) * 0.045
    if policy_type == "health":
        return 1 + max(0, age - 18) * 0.02
    if age < 25:
        return 1.35
    return 1.0 if age < 65 else 1.1


def premium(plan: Plan, age: int, coverage: int, deductible: int) -> float:
    raw: float = (
        plan["rate_per_1000"]
        * coverage
        / 1000
        * age_factor(plan["policy_type"], age)
        * DEDUCTIBLE_FACTORS[deductible]
        + plan["annual_fee"]
    )
    return round(raw, 2)


def search_policies(policy_type: str) -> dict[str, Any]:
    if policy_type not in POLICY_TYPES:
        return {"error": f"Unknown policy_type {policy_type!r}. Valid: {', '.join(POLICY_TYPES)}"}
    keys = ("plan_id", "insurer", "plan_name", "min_coverage", "max_coverage")
    return {
        "plans": [{k: p[k] for k in keys} for p in load_catalog() if p["policy_type"] == policy_type]
    }


def get_quote(plan_id: str, age: int, coverage_amount: int, deductible: int) -> dict[str, Any]:
    plan = next((p for p in load_catalog() if p["plan_id"] == plan_id), None)
    if plan is None:
        return {"error": f"Unknown plan_id {plan_id!r}"}
    if not plan["min_coverage"] <= coverage_amount <= plan["max_coverage"]:
        return {
            "error": f"coverage_amount must be between {plan['min_coverage']} and {plan['max_coverage']}"
        }
    if deductible not in DEDUCTIBLE_FACTORS:
        return {"error": f"Invalid deductible {deductible}. Valid: {sorted(DEDUCTIBLE_FACTORS)}"}
    return {
        "plan_id": plan_id,
        "plan_name": plan["plan_name"],
        "insurer": plan["insurer"],
        "annual_premium_usd": premium(plan, age, coverage_amount, deductible),
    }
