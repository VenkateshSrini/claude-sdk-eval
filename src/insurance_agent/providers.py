from strands.models import Model
from strands.models.anthropic import AnthropicModel
from strands.models.bedrock import BedrockModel

from insurance_agent.config import required


def _provider() -> str:
    provider = required("LLM_PROVIDER")
    if provider not in ("anthropic", "bedrock"):
        raise RuntimeError(f"Unsupported LLM_PROVIDER {provider!r}. Use 'anthropic' or 'bedrock'.")
    return provider


def agent_env() -> dict[str, str]:
    """Environment for the Claude Code CLI subprocess spawned by the Agent SDK."""
    if _provider() == "bedrock":
        return {"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": required("AWS_REGION")}
    # "0" overrides a stray CLAUDE_CODE_USE_BEDROCK=1 inherited from the shell (see NOTES.md).
    return {"ANTHROPIC_API_KEY": required("ANTHROPIC_API_KEY"), "CLAUDE_CODE_USE_BEDROCK": "0"}


def judge_model() -> Model:
    model_id = required("JUDGE_MODEL")
    if _provider() == "bedrock":
        return BedrockModel(model_id=model_id, region_name=required("AWS_REGION"))
    return AnthropicModel(
        client_args={"api_key": required("ANTHROPIC_API_KEY")}, model_id=model_id, max_tokens=2048
    )


def describe() -> str:
    provider = _provider()
    region = required("AWS_REGION") if provider == "bedrock" else "n/a"
    return (
        f"provider={provider} agent_model={required('AGENT_MODEL')} "
        f"judge_model={required('JUDGE_MODEL')} region={region}"
    )
