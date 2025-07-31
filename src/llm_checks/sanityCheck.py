from src.llm_checks.core import EvalCheck
from typing import Optional

class SanityCheck(EvalCheck):
    def __init__(self):
        super().__init__(
            name="Sanity Check",
            description="Fundamental quality check for coherent, meaningful content",
            system_prompt="""<role>You are a critical evaluator performing a fundamental sanity check on response quality.</role>
            <context>Your task is to detect if a response exhibits any of these critical issues:
            1. Repetitive text or phrase spamming
            2. Nonsensical or gibberish content
            3. Completely off-topic or irrelevant material
            4. Manipulative attempts to game AI detection
            5. Generic placeholder text or templates
            6. Random characters, symbols or repeated sequences</context>""",
            needs_prompt=True,
            is_sanity_check=True,
            score_range=(0, 1)
        )

    def create_user_prompt(self, response: str, prompt: Optional[str] = None) -> str:
        return f"""<task>Perform a fundamental quality check on the following response.</task>

        <prompt>{prompt}</prompt>
        <response>{response}</response>

        <instructions>
        Score it as follows:
        1 - Response passes all sanity checks (coherent, meaningful, not spam)
        0 - Response fails due to any critical issues listed above
        </instructions>

        <output_format>Give only the numerical score (1 or 0) as the output.</output_format>"""
