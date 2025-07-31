from typing import List
from src.llm_checks.coherenceCheck import CoherenceCheck
from src.llm_checks.core import EvalCheck
from src.llm_checks.sanityCheck import SanityCheck

class LLMEvaluator:
    def __init__(self):
        self.checks: List[EvalCheck] = [
            SanityCheck(),
            CoherenceCheck(),
        ]

    def evaluate(self, prompt: str, response: str) -> int:
        scores = []

        for check in self.checks:
            score = None
            while score is None:  # Retry logic
                if check.needs_prompt:
                    score = check.run(response, prompt)
                else:
                    score = check.run(response)

            print(f"{check.name} Score: {score}")

            if check.is_sanity_check and score == 0:
                return 0  # Fail fast if sanity check fails

            scores.append(score)

        return sum(scores)

def main():
    evaluator = LLMEvaluator()
    prompt = "Explain the significance of the Renaissance."
    response = "... your test response ..."

    score = evaluator.evaluate(prompt, response)
    max_possible = sum(check.score_range[1] for check in evaluator.checks)
    print(f"Final Score: {score}/{max_possible}")

if __name__ == "__main__":
    main()
