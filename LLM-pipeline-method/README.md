# Formal Specification Generation Pipeline

A sophisticated automated pipeline for generating and validating formal specifications using Large Language Models (LLMs) with a quality feedback loop.

## Pipeline Overview

```
Ground Truth Summary
      ↓
   [LLM1] → Generate Formal Specification
      ↓
   [LLM2] → Generate Summary from Spec
      ↓
   [LLM3] → Judge Similarity (vs Ground Truth)
      ↓
   Is Similarity ≥ Threshold?
      ├─ YES → Success! Return Formal Spec
      └─ NO  → Loop back to LLM1 (max iterations)
```

## Features

- **Automated Feedback Loop**: Iteratively refines formal specifications until quality threshold is met
- **Multiple LLM Providers**: Support for OpenAI, Anthropic, and mock LLMs
- **Flexible Similarity Checking**: TF-IDF, semantic embeddings, or LLM-based judge
- **Comprehensive Logging**: Detailed tracking of each iteration
- **Result Persistence**: Save pipeline results to JSON for analysis
- **Configurable**: All parameters via environment variables or config file

## Project Structure

```
.
├── config.py                 # Configuration management
├── llm_client.py            # LLM provider abstractions
├── similarity_checker.py     # Similarity metrics
├── pipeline.py              # Main pipeline orchestrator
├── logger.py                # Logging utility
├── main.py                  # Entry point
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variables template
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
cd /home/chandrima/Desktop/Academics/sem8/Formal-Spec-Gen
```

2. Create a virtual environment (optional but recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

### Environment Variables

Create a `.env` file with the following (optional - uses defaults if not specified):

```
# LLAMA API Configuration
LLAMA_API_KEY=your_llama_api_key_here

# LLM Model Configuration (LLAMA models)
LLM1_MODEL=meta-llama/Llama-2-70b-chat-hf      # Spec generation (default)
LLM2_MODEL=meta-llama/Llama-2-13b-chat-hf      # Summary generation (default)
LLM3_MODEL=meta-llama/Llama-2-7b-chat-hf       # Judge (default)

# Temperature values (0.0-2.0)
LLM1_TEMPERATURE=0.7                # Higher = more creative specs
LLM2_TEMPERATURE=0.5                # Balanced summary generation
LLM3_TEMPERATURE=0.0                # Deterministic judging

# Pipeline Parameters
SIMILARITY_THRESHOLD=0.85           # Similarity score threshold (0-1)
MAX_ITERATIONS=5                    # Maximum retry iterations
VERBOSE=true                        # Enable verbose logging
```

## Usage

### Quick Start with Mock LLMs (No API Keys Needed)

```bash
python main.py --mode mock
```

### With Real LLMs (LLAMA)

First, set your LLAMA API key:
```bash
export LLAMA_API_KEY="your_key_here"
```

Then run:
```bash
python main.py --mode real
```

### Programmatic Usage with LLAMA

```python
from config import create_default_config
from pipeline import FormalSpecGenerationPipeline

# Create configuration (loads LLAMA_API_KEY from environment)
config = create_default_config()

# Create pipeline with LLAMA provider
pipeline = FormalSpecGenerationPipeline(
    config,
    use_llm_judge=True,
    similarity_method="llm",
    llm_provider="llama"  # Use LLAMA as provider
)

# Define ground truth summary
ground_truth = """
The system must process 1000 requests per second.
All data must be encrypted with AES-256.
System availability must be >= 99.99%.
"""

# Run pipeline
result = pipeline.run(ground_truth)

# Print results
pipeline.print_summary(result)

# Save results
pipeline.save_results(result, output_file="my_results.json")
```

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --mode {mock, real}              Run mode (default: mock)
  --summary TEXT                   Custom ground truth summary
  --threshold FLOAT                Similarity threshold (0-1)
  --max-iterations INT             Maximum iterations
  --output FILE                    Output file path
  --similarity-method {tfidf, semantic, llm}
                                   Similarity checking method
  --llm-provider {openai, anthropic, llama}
                                   LLM provider to use (default: llama)
```

## How It Works

### Step 1: Formal Specification Generation (LLM1)
- Takes a natural language summary as input
- Generates formal specification using mathematical notation
- Can use Z notation, TLA+, or other formal methods

### Step 2: Summary Generation (LLM2)
- Takes the formal specification
- Generates a concise natural language summary
- Captures key requirements and properties

### Step 3: Quality Check (LLM3/Similarity Metric)
- Compares original summary with generated summary
- Calculates similarity score (0-1)
- Determines if specifications meet quality threshold

### Step 4: Feedback Loop
- If similarity >= threshold: Success!
- If similarity < threshold: Return to Step 1 with improved context
- Continue until threshold is met or max iterations reached

## Output Files

The pipeline generates:

1. **JSON Results File** (`pipeline_results.json`):
```json
{
  "success": true,
  "ground_truth_summary": "...",
  "final_formal_spec": "...",
  "final_summary": "...",
  "final_similarity_score": 0.92,
  "total_iterations": 2,
  "timestamp": "2026-03-10T10:30:45.123456"
}
```

2. **Console Logs**: Real-time progress and iteration details

## Customization

### Custom LLM Provider

Extend the `LLMClient` class:

```python
from llm_client import LLMClient

class MyCustomLLM(LLMClient):
    def generate(self, prompt: str) -> str:
        # Your implementation
        pass
    
    def generate_number(self, prompt: str) -> float:
        # Your implementation
        pass
```

### Custom Similarity Metric

Extend the `SimilarityChecker` class:

```python
from similarity_checker import SimilarityChecker

class MyCustomSimilarity(SimilarityChecker):
    def calculate_similarity(self, text1: str, text2: str) -> float:
        # Your implementation
        pass
```

### Custom Prompts

Modify the prompt templates in `config.py`:

```python
config.formal_spec_prompt_template = """Your custom prompt here"""
config.summary_from_spec_prompt_template = """Your custom prompt here"""
config.judge_prompt_template = """Your custom prompt here"""
```

## Similarity Methods

### TF-IDF (Default)
- Fast, no external dependencies
- Good for keyword-based similarity
- Best for domain-specific terminology

### Semantic Similarity
- Uses sentence embeddings
- Requires: `pip install sentence-transformers`
- Better captures semantic meaning
- Slower but more accurate

### LLM Judge (Advanced)
- Uses LLM3 to judge similarity
- Most flexible and powerful
- Requires LLM API calls (slower, higher cost)
- Best for nuanced semantic understanding

## Performance Tips

1. **Use TF-IDF** for fastest execution (no API calls)
2. **Set higher temperature for LLM1** for more diverse specs
3. **Start with high threshold** and lower if needed
4. **Use shorter prompts** for faster LLM responses
5. **Batch process** multiple summaries in parallel

## Troubleshooting

### "API key not found"
```bash
export OPENAI_API_KEY="your_key_here"
```

### High iteration count
- Lower the similarity threshold
- Use more detailed ground truth summary
- Try semantic similarity instead of TF-IDF

### LLM errors
- Check API rate limits
- Verify API key is valid
- Check network connectivity

## Future Enhancements

- [ ] Multi-threaded pipeline for parallel processing
- [ ] Support for more LLM providers (Claude, Llama, etc.)
- [ ] Advanced similarity metrics (BLEU, ROUGE, etc.)
- [ ] Specification quality metrics
- [ ] Visualization dashboard
- [ ] Caching layer for repeated summaries
- [ ] Integration with spec verification tools

## License

MIT

## Author

Formal Specification Research Team
