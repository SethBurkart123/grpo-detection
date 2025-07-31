# GPRO undetectable AI transformer model

Building an undetectable model using GRPO RL.

## Function docs

### Detect AI

```python
from src.detection import detect

# Example usage
text = "Your text to analyze here"
scores = detect(text)

# Sample response
{
    'scoreGptZero': 50,      # GPT-Zero detection score
    'scoreOpenAI': 0,        # OpenAI detection score
    'scoreWriter': 0,        # Writer.com detection score
    'scoreCrossPlag': 0,     # Cross-plagiarism score
    'scoreCopyLeaks': 100,   # CopyLeaks detection score
    'scoreSapling': 50,      # Sapling detection score
    'scoreContentAtScale': 50, # ContentAtScale detection score
    'scoreZeroGPT': 50,      # ZeroGPT detection score
    'human': 50              # Human writing probability
}
```

Each score ranges from 0 to 100, where higher values indicate a stronger likelihood of AI-generated content.

#### Error Handling

The function returns an empty dictionary `{}` if any error occurs during the detection process.