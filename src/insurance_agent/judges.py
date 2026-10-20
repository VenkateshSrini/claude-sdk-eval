from strands import Agent
from strands.models import Model
from strands_evals.evaluators import Evaluator
from strands_evals.types import EvaluationData, EvaluationOutput


class RubricJudge(Evaluator[str, str]):
    """One LLM-as-judge class; each instance is named for the report (`name=`)."""

    def __init__(
        self, name: str, rubric: str, model: Model, contexts: dict[str, str] | None = None
    ) -> None:
        super().__init__(name=name)
        self.rubric = rubric
        self.model = model
        self.contexts = contexts

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        parts = [
            f"<Input>{evaluation_case.input}</Input>",
            f"<ExpectedOutput>{evaluation_case.expected_output}</ExpectedOutput>",
            f"<ActualOutput>{evaluation_case.actual_output}</ActualOutput>",
        ]
        if self.contexts is not None:
            context = self.contexts.get(evaluation_case.input)
            if not context:
                raise RuntimeError(f"No retrieved context stored for case input: {evaluation_case.input!r}")
            parts.append(f"<RetrievedContext>{context}</RetrievedContext>")
        judge = Agent(model=self.model, system_prompt=self.rubric, callback_handler=None)
        result = judge("\n".join(parts), structured_output_model=EvaluationOutput)
        if result.structured_output is None:
            raise RuntimeError("Judge returned no structured output")
        return [result.structured_output]
