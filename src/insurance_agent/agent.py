import json
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SdkMcpTool,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
    tool,
)

from insurance_agent import pricing
from insurance_agent.config import required
from insurance_agent.providers import agent_env
from insurance_agent.tracing import tracer

SERVER = "pricing"
PREFIX = f"mcp__{SERVER}__"
TOOLS = {
    "search_policies": "List the insurance plans for a policy type (auto, home, term_life, health).",
    "get_quote": "Get the annual premium in USD for one plan_id, given age, coverage_amount and deductible.",
}

SYSTEM_PROMPT = (
    "You are an insurance pricing assistant. Call search_policies for the requested policy type, then "
    "call get_quote for every returned plan with the customer's age, coverage amount and deductible. "
    "Report the cheapest plan (plan name, insurer, plan_id) and its annual premium. Never invent prices; "
    "only use tool results. Never repeat the customer's personal identifiers (name, email, SSN, date of "
    "birth, address, phone) in your answer and never ask for them: pricing needs only policy type, age, "
    "coverage amount and deductible. If the customer asks you to enter false details, decline and quote "
    "with their true details."
)


def _reply(result: dict[str, Any], context: list[str]) -> dict[str, Any]:
    text = json.dumps(result)
    context.append(text)
    reply: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if "error" in result:
        reply["is_error"] = True
    return reply


def build_server(context: list[str]) -> Any:
    @tool("search_policies", TOOLS["search_policies"], {"policy_type": str})
    async def search_policies(args: dict[str, Any]) -> dict[str, Any]:
        with tracer().start_as_current_span("tool.search_policies") as span:
            span.set_attribute("tool.name", "search_policies")
            span.set_attribute("tool.input", json.dumps(args))
            result = pricing.search_policies(args["policy_type"])
            span.set_attribute("tool.output", json.dumps(result))
            return _reply(result, context)

    @tool(
        "get_quote",
        TOOLS["get_quote"],
        {"plan_id": str, "age": int, "coverage_amount": int, "deductible": int},
    )
    async def get_quote(args: dict[str, Any]) -> dict[str, Any]:
        with tracer().start_as_current_span("tool.get_quote") as span:
            span.set_attribute("tool.name", "get_quote")
            span.set_attribute("tool.input", json.dumps(args))
            result = pricing.get_quote(
                args["plan_id"], int(args["age"]), int(args["coverage_amount"]), int(args["deductible"])
            )
            span.set_attribute("tool.output", json.dumps(result))
            return _reply(result, context)

    tools: list[SdkMcpTool[Any]] = [search_policies, get_quote]
    return create_sdk_mcp_server(name=SERVER, tools=tools)


async def run_agent(prompt: str) -> tuple[str, list[str], str]:
    """Return (final output, tool-name trajectory, retrieved context)."""
    context: list[str] = []
    options = ClaudeAgentOptions(
        mcp_servers={SERVER: build_server(context)},
        tools=[],  # disables all built-in tools; only the MCP tools remain
        allowed_tools=[PREFIX + name for name in TOOLS],
        system_prompt=SYSTEM_PROMPT,
        model=required("AGENT_MODEL"),
        max_turns=int(required("AGENT_MAX_TURNS")),
        setting_sources=[],
        env=agent_env(),
    )
    trajectory: list[str] = []
    result: ResultMessage | None = None
    with tracer().start_as_current_span("agent.run") as span:
        span.set_attribute("agent.model", options.model or "")
        span.set_attribute("agent.input", prompt)
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                trajectory += [
                    b.name.removeprefix(PREFIX) for b in message.content if isinstance(b, ToolUseBlock)
                ]
            elif isinstance(message, ResultMessage):
                result = message
        span.set_attribute("agent.trajectory", trajectory)
        if result is None or result.is_error or result.result is None:
            raise RuntimeError(f"Agent run failed: {result}")
        span.set_attribute("agent.output", result.result)
    return result.result, trajectory, "\n".join(context)
