import ollama


def run_prompt(system_prompt, user_prompt, temperature=0.7, max_tokens=128):
    """
    Run a prompt using the provided system and user inputs.
    
    Args:
        system_prompt (str): The system instruction to set the context.
        user_prompt (str): The user query.
        temperature (float): Sampling temperature for the response (0.0 to 1.0).
        max_tokens (int): Maximum number of tokens to generate.
    
    Returns:
        str: The generated output from the model.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Call Ollama's chat function with the messages and generation parameters.
    response = ollama.chat(
        model='dolphin3',
        messages=messages,
        options={
            "num_predict": max_tokens,
            "temperature": temperature,
        }
    )
    
    # Extract and return the assistant's reply.
    return response['message']['content']

# Example usage:
if __name__ == "__main__":
    # Initialize your model (for example, "mistral")

    # Define system and user prompts
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is the capital of France? Make it funny"

    # Run the prompt with specific temperature and max tokens settings
    output = run_prompt(system_prompt, user_prompt, temperature=1, max_tokens=50)
    print("Output:", output)
