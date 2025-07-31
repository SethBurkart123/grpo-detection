from src.llm_checks.core import EvalCheck
from typing import Optional

class CoherenceCheck(EvalCheck):
    def __init__(self):
        super().__init__(
            name="Coherence",
            description="Evaluates logical flow and consistency",
            system_prompt="""You are an expert evaluator assessing the logical coherence of a response.
            A perfect response (score = 10) has a well-structured flow, clear logical transitions, and no contradictions.
            A low score (0-3) means the response is disjointed, confusing, or internally inconsistent."""
        )

    def create_user_prompt(self, response: str, prompt: Optional[str] = None) -> str:
        return f"""Evaluate the following response for coherence and logical consistency.

        Response: {response}

        Score it from 0-10, where:
        10 - Fully coherent and structured
        7-9 - Mostly logical, minor inconsistencies
        4-6 - Somewhat disjointed, but understandable
        1-3 - Lacks structure, hard to follow
        0 - Completely incoherent

        Give only the numerical score as the output."""
