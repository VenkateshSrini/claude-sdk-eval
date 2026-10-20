from pathlib import Path
from typing import IO

from arize.otel import Endpoint, Transport, register
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from insurance_agent.config import required

SERVICE = "insurance-pricing-agent"
SPAN_KIND = "openinference.span.kind"  # Arize/OpenInference attribute; values "AGENT", "TOOL"


def tracer() -> trace.Tracer:
    return trace.get_tracer(SERVICE)


class AgentIdentityProcessor(SpanProcessor):
    """Stamp AGENT_ID and AGENT_VERSION on every span. Read from the environment at start-up, so a
    deployment that sets different values in its environment replaces the ones in .env."""

    def __init__(self) -> None:
        self._attributes = {
            "agent.id": required("AGENT_ID"),
            "gen_ai.agent.id": required("AGENT_ID"),
            "agent.version": required("AGENT_VERSION"),
        }

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        span.set_attributes(self._attributes)

    def on_end(self, span: ReadableSpan) -> None:
        pass


def setup_tracing(file_path: Path) -> tuple[TracerProvider, IO[str]]:
    """Install a global tracer provider: identity stamp, then a JSON-per-line file, then Arize."""
    handle = file_path.open("w", encoding="utf-8")
    exporter = ConsoleSpanExporter(out=handle, formatter=lambda span: span.to_json(indent=None) + "\n")
    provider = register(
        space_id=required("ARIZE_SPACE_ID"),
        api_key=required("ARIZE_API_KEY"),
        project_name=required("ARIZE_PROJECT_NAME"),
        endpoint=Endpoint.ARIZE.value,  # plain str: the enum itself stringifies as 'Endpoint.ARIZE' on Python 3.15
        transport=Transport.HTTP,
        span_processors=[AgentIdentityProcessor(), SimpleSpanProcessor(exporter)],
        verbose=False,
    )
    return provider, handle
