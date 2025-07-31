import requests
import json
import random
import time
from typing import Dict, Union, Tuple
import os

class UserAgentManager:
    """
    Manages user agent and platform headers to ensure consistent browser fingerprinting.
    Each user agent is paired with its corresponding platform for authenticity.
    """

    _USER_AGENT_PLATFORMS: list[Tuple[str, str]] = [
        (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
            'macOS'
        ),
        (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
            'Windows'
        ),
        (
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
            'Linux'
        ),
        (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/138.0.0.0 Safari/537.36',
            'Windows'
        )
    ]

    @classmethod
    def get_submission_headers(cls) -> Dict[str, str]:
        """
        Returns headers for the initial submission request to undetectable.ai
        
        Returns:
            Dictionary of HTTP headers for submission
        """
        user_agent, platform = random.choice(cls._USER_AGENT_PLATFORMS)

        return {
            'accept': 'text/x-component',
            'accept-language': 'en-GB,en;q=0.9',
            'cache-control': 'no-cache',
            'content-type': 'text/plain;charset=UTF-8',
            'dnt': '1',
            'next-action': '8b888df218472b367d6709b65423720937e55d44',
            'next-router-state-tree': '%5B%22%22%2C%7B%22children%22%3A%5B%5B%22locale%22%2C%22en%22%2C%22d%22%5D%2C%7B%22children%22%3A%5B%22__PAGE__%22%2C%7B%7D%2C%22%2F%22%2C%22refresh%22%5D%7D%5D%7D%2Cnull%2Cnull%2Ctrue%5D',
            'origin': 'https://undetectable.ai',
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'referer': 'https://undetectable.ai/',
            'sec-ch-ua': '"Not)A;Brand";v="8", "Chromium";v="138", "Brave";v="138"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': f'"{platform}"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'sec-gpc': '1',
            'user-agent': user_agent,
        }

    @classmethod
    def get_query_headers(cls) -> Dict[str, str]:
        """
        Returns headers for the query request to get results
        
        Returns:
            Dictionary of HTTP headers for querying results
        """
        user_agent, platform = random.choice(cls._USER_AGENT_PLATFORMS)

        return {
            'accept': 'application/json',
            'accept-language': 'en-GB,en;q=0.9',
            'cache-control': 'no-cache',
            'content-type': 'application/json',
            'dnt': '1',
            'origin': 'https://undetectable.ai',
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'referer': 'https://undetectable.ai/',
            'sec-ch-ua': '"Not)A;Brand";v="8", "Chromium";v="138", "Brave";v="138"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': f'"{platform}"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'cross-site',
            'sec-gpc': '1',
            'user-agent': user_agent,
        }

def parse_submission_response(response_text: str) -> str:
    """
    Parse the submission response to extract the task ID
    
    Args:
        response_text: Raw response text from the submission API
        
    Returns:
        Task ID string, or empty string if parsing fails
    """
    try:
        lines = response_text.strip().split('\n')
        for line in lines:
            if line.startswith('1:'):
                # Extract the JSON part after "1:"
                json_str = line[2:]
                data = json.loads(json_str)
                return data.get('id', '')
    except Exception as e:
        print(f"Error parsing submission response: {e}")
    return ''

def detect(prompt: str, log_file_path: str) -> dict:
    """
    Detect AI-generated content using undetectable.ai's new two-step API and log results to a JSONL file.
    
    Args:
        prompt: Text to analyze.
        log_file_path: Path to the JSONL log file.
        
    Returns:
        Dictionary containing the detection results.
    """
    
    # Step 1: Submit the text for analysis
    submission_data = json.dumps([prompt, "l6_v6", False])
    
    try:
        # Submit text for analysis
        submission_headers = UserAgentManager.get_submission_headers()
        submission_response = requests.post(
            'https://undetectable.ai/', 
            headers=submission_headers, 
            data=submission_data
        )
        submission_response.raise_for_status()
        
        # Extract task ID from response
        task_id = parse_submission_response(submission_response.text)
        if not task_id:
            raise Exception("Failed to extract task ID from submission response")
        
        print(f"Submitted text for analysis, task ID: {task_id}")
        
        # Step 2: Poll for results
        query_headers = UserAgentManager.get_query_headers()
        query_data = json.dumps({"id": task_id})
        
        max_attempts = 30  # Maximum polling attempts
        for attempt in range(max_attempts):
            time.sleep(2)  # Wait before polling
            
            query_response = requests.post(
                'https://sea-lion-app-3p5x4.ondigitalocean.app/query',
                headers=query_headers,
                data=query_data
            )
            query_response.raise_for_status()
            
            result_data = query_response.json()
            status = result_data.get('status')
            
            if status == 'done':
                # Analysis complete, extract results
                result_score = result_data.get('result', 0)
                result_details = result_data.get('result_details', {})
                result_categories = result_data.get('result_categories', {})
                
                average_score = result_score / 100.0  # Convert to 0-1 scale
                
                log_entry = {
                    "prompt": prompt,
                    "task_id": task_id,
                    "result_score": result_score,
                    "result_details": result_details,
                    "result_categories": result_categories,
                    "average_score": average_score
                }
                
                with open(log_file_path, "a") as log_file:
                    json.dump(log_entry, log_file)
                    log_file.write("\n")
                
                return {"score": average_score, "details": result_details}
                
            elif status == 'pending':
                print(f"Analysis still pending, attempt {attempt + 1}/{max_attempts}")
                continue
            else:
                raise Exception(f"Unexpected status: {status}")
        
        # If we get here, we exceeded max attempts
        raise Exception("Analysis timed out after maximum polling attempts")
        
    except requests.RequestException as e:
        print(f"Request error: {e}")
        log_entry = {"prompt": prompt, "error": str(e)}
        with open(log_file_path, "a") as log_file:
            json.dump(log_entry, log_file)
            log_file.write("\n")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        log_entry = {"prompt": prompt, "error": str(e)}
        with open(log_file_path, "a") as log_file:
            json.dump(log_entry, log_file)
            log_file.write("\n")
        return {}

def run_detection(prompts: list[str]):
    """
    Runs AI detection on a list of prompts and logs the results.

    Args:
       prompts: A list of text prompts to analyze.
    """

    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Create a unique log file for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file_path = os.path.join(logs_dir, f"detection_log_{timestamp}.jsonl")

    for prompt in prompts:
        print(f"Analyzing: {prompt[:50]}...")
        result = detect(prompt, log_file_path)
        if result:
            print(f"Score: {result.get('score', 0):.3f}")
        print("-" * 50)

if __name__ == '__main__':
    # Example Usage
    example_prompts = [
        "This is a test of the emergency broadcast system.",
        "The quick brown fox jumps over the lazy dog.",
        "According to all known laws of aviation, there is no way that a bee should be able to fly.",
        "Hello, world!"
    ]

    run_detection(example_prompts)
