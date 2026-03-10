# LLAMA API Setup Guide

This guide explains how to set up and use LLAMA with the Formal Spec Generation Pipeline.

## Supported LLAMA Providers

### 1. Together AI (Recommended)
Together AI provides hosted LLAMA models with a simple API.

**Setup:**
1. Create account at https://www.together.ai/
2. Get your API key from dashboard
3. Set environment variable:
```bash
export LLAMA_API_KEY="your_together_api_key_here"
```

**Default models:**
- LLM1: `meta-llama/Llama-2-70b-chat-hf` (Spec generation)
- LLM2: `meta-llama/Llama-2-13b-chat-hf` (Summary generation)  
- LLM3: `meta-llama/Llama-2-7b-chat-hf` (Judge/comparison)

**Usage:**
```bash
python main.py --mode real --llm-provider llama
```

### 2. Replicate
Alternative LLAMA hosting provider.

**Setup:**
1. Create account at https://replicate.com/
2. Get your API key
3. Configure in code or environment

### 3. Local LLAMA (Ollama)
Run LLAMA locally with Ollama.

**Setup:**
1. Install Ollama from https://ollama.ai/
2. Pull model: `ollama pull llama2`
3. Run server: `ollama serve`
4. Update base URL in configuration

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Create `.env` file:
```bash
cp .env.example .env
```

3. Edit `.env` and add your LLAMA API key:
```
LLAMA_API_KEY=your_key_here
```

## Quick Start

### Test Connection
```python
from config import LLMConfig
from llm_client import LlamaClient

config = LLMConfig(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    api_key="your_api_key",
)
client = LlamaClient(config)
response = client.generate("Hello, how are you?")
print(response)
```

### Run Pipeline
```bash
# With mock LLMs (no API key needed)
python main.py --mode mock

# With real LLAMA models
export LLAMA_API_KEY="your_key_here"
python main.py --mode real

# Run tests
python test_pipeline.py
```

## Pricing

### Together AI
- Llama-2 70b: ~$0.90/1M tokens
- Llama-2 13b: ~$0.22/1M tokens
- Llama-2 7b: ~$0.10/1M tokens

Estimated cost per pipeline run: $0.01 - $0.10 (depending on model size and iterations)

## Configuration

### Model Selection
Change models by setting environment variables:
```bash
export LLM1_MODEL="meta-llama/Llama-2-70b-chat-hf"
export LLM2_MODEL="meta-llama/Llama-2-13b-chat-hf"
export LLM3_MODEL="meta-llama/Llama-2-7b-chat-hf"
```

### Temperature
Adjust model creativity:
```bash
export LLM1_TEMPERATURE=0.7  # More creative
export LLM2_TEMPERATURE=0.5  # Balanced
export LLM3_TEMPERATURE=0.0  # Deterministic
```

### API Endpoint
For custom endpoints (e.g., self-hosted):
```python
from config import LLMConfig
from llm_client import LlamaClient

config = LLMConfig(
    model_name="llama2",
    api_key="your_key"
)
client = LlamaClient(config, base_url="http://localhost:8000/v1")
```

## Troubleshooting

### "Invalid API key"
- Verify key in Together AI dashboard
- Check key is correctly set: `echo $LLAMA_API_KEY`
- Restart terminal after setting environment variable

### Rate limit exceeded
- Lower `max_iterations` in `.env`
- Use smaller models (7b instead of 70b)
- Increase requests delay

### Out of memory
- Use smaller models (7b instead of 13b/70b)
- Reduce `max_tokens`

### Connection timeout
- Check internet connection
- Increase `timeout` in LLMConfig
- Try different model

## Advanced Usage

### Switch Providers Dynamically
```python
from pipeline import FormalSpecGenerationPipeline
from config import create_default_config

config = create_default_config()

# Use LLAMA
pipeline_llama = FormalSpecGenerationPipeline(config, llm_provider="llama")

# Use OpenAI
pipeline_openai = FormalSpecGenerationPipeline(config, llm_provider="openai")
```

### Custom Base URL
```python
from llm_client import LlamaClient

client = LlamaClient(
    config, 
    base_url="https://custom-endpoint.com/v1"
)
```

## Performance Tips

1. **Use 7b model for judge (LLM3)** - Fast and accurate for similarity
2. **Use 70b model for spec generation** - Most accurate specifications
3. **Use 13b model for summarization** - Good balance of speed/quality
4. **Start with low iterations** - Adjust threshold instead
5. **Cache results** - Reuse results for repeated summaries

## Supported LLAMA Models

From Together AI:
- `meta-llama/Llama-2-7b-chat-hf` ✓ Recommended for judge
- `meta-llama/Llama-2-13b-chat-hf` ✓ Good balance
- `meta-llama/Llama-2-70b-chat-hf` ✓ Best quality
- `meta-llama/Code-Llama-13b-chat-hf` - Code-focused
- `meta-llama/Code-Llama-34b-chat-hf` - Code-focused, larger

## Contact Support

For issues with:
- **Together AI**: https://www.together.ai/support
- **Pipeline code**: Open issue on GitHub
- **LLAMA models**: https://github.com/meta-llama/llama

## Additional Resources

- LLAMA Model Card: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
- Together AI Docs: https://docs.together.ai/
- OpenAI Compatibility: The LLAMA client uses OpenAI-compatible API format
