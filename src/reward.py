from concurrent.futures import ThreadPoolExecutor
import time
import os

from src.llm_eval import LLMEvaluator
from src.detection import detect

logs_dir = "logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

llm_evaluator = LLMEvaluator()

# Create a unique log file for this run
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_file_path = os.path.join(logs_dir, f"detection_log_{timestamp}.jsonl")

def get_detection_score(comp, log_file_path):
  return (detect(comp, log_file_path).get("score", 0) * 2) ** 2

def get_llm_score(prompt, comp):
  score = llm_evaluator.evaluate(prompt, comp)
  max_possible = sum(check.score_range[1] for check in llm_evaluator.checks)
  return score / max_possible

def parallel_scoring(prompt, comp, log_file_path):
  """Run detection and LLM scoring in parallel and return combined score."""
  with ThreadPoolExecutor(max_workers=2) as executor:
    # Submit both tasks
    detection_future = executor.submit(get_detection_score, comp, log_file_path)
    #llm_future = executor.submit(get_llm_score, prompt, comp)

    # Get results
    # 0-4
    detection_score = detection_future.result()
    # 0-1
    #llm_score = llm_future.result()

    #print(f"Detection score: {detection_score} --- LLM score: {llm_score}", end="--- ")
    print(f"Detection score: {detection_score}")

    # Combine scores
    #return (detection_score * llm_score) / 4
    return detection_score / 4

# Then modify your custom_reward_fn to use this:
def custom_reward_fn(prompts, completions, **kwargs):
  """Custom reward function that runs detection and LLM scoring in parallel."""
  rewards = []
  print("")
  for prompt, comp in zip(prompts, completions):
    score = parallel_scoring(prompt, comp, log_file_path)
    print(f"Score: {score}")
    rewards.append(score)
  return rewards
