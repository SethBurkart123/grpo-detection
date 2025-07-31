import time
import os
from ripplex import flow, loop

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

@flow
def parallel_scoring(prompt, comp, log_file_path):
  """Run detection and LLM scoring in parallel and return combined score."""
  # These run in PARALLEL automatically with @flow
  detection_score = get_detection_score(comp, log_file_path)  # 0-4
  #llm_score = get_llm_score(prompt, comp)  # 0-1

  #print(f"Detection score: {detection_score} --- LLM score: {llm_score}", end="--- ")
  print(f"Detection score: {detection_score}")

  # Combine scores
  #return (detection_score * llm_score) / 4
  return detection_score / 4

def custom_reward_fn(prompts, completions, **kwargs):
  """Custom reward function that runs detection and LLM scoring in parallel."""
  print("")
  
  # Process all prompt/completion pairs in parallel
  @loop(zip(prompts, completions))
  def score_pair(prompt_comp):
    prompt, comp = prompt_comp
    # log_file_path is automatically captured from outer scope
    score = parallel_scoring(prompt, comp, log_file_path)
    print(f"Score: {score}")
    return score
  
  return score_pair
