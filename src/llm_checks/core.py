from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod
from src.ollama_generate import run_prompt

def parse_score(output: str, min_score: int = 0, max_score: int = 10) -> Optional[int]:
    """Extracts the first numeric score from the LLM output."""
    try:
        score = int(output.strip().split()[0])
        if min_score <= score <= max_score:
            return score
    except ValueError:
        pass
    return None

@dataclass
class EvalCheck(ABC):
    name: str
    description: str
    system_prompt: str
    needs_prompt: bool = False
    is_sanity_check: bool = False
    score_range: tuple = (0, 10)

    @abstractmethod
    def create_user_prompt(self, response: str, prompt: Optional[str] = None) -> str:
        """Create the user prompt for evaluation."""
        pass

    def run(self, response: str, prompt: Optional[str] = None) -> Optional[int]:
        user_prompt = self.create_user_prompt(response, prompt)
        output = run_prompt(self.system_prompt, user_prompt, temperature=1, max_tokens=2)
        return parse_score(output, *self.score_range)
