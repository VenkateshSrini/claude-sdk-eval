from pathlib import Path
from typing import IO

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

SERVICE = "insurance-pricing-agent"


def tracer() -> trace.Tracer:
    return trace.get_tracer(SERVICE)


def setup_tracing(file_path: Path) -> tuple[TracerProvider, IO[str]]:
    """Install a global tracer provider that writes one JSON span per line to file_path."""
    handle = file_path.open("w", encoding="utf-8")
    exporter = ConsoleSpanExporter(out=handle, formatter=lambda span: span.to_json(indent=None) + "\n")
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider, handle
