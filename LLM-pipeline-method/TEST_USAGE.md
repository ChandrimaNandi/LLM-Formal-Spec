# Using test_with_summaries.py

This script allows you to run the Formal Spec Generation Pipeline on multiple summaries loaded from a JSON file.

## Quick Start

### 1. Prepare Your Summaries

Create or edit `summary.json` with your summaries:

```json
{
  "summaries": [
    {
      "name": "my_summary_1",
      "content": "The system must..."
    },
    {
      "name": "my_summary_2", 
      "content": "The authentication system..."
    }
  ]
}
```

### 2. Run With Mock LLMs (No API Keys)

```bash
python test_with_summaries.py --mock
```

### 3. Run With Real LLAMA

```bash
export LLAMA_API_KEY="your_key_here"
python test_with_summaries.py --llm-provider llama
```

## Command Line Options

```bash
python test_with_summaries.py [OPTIONS]

Options:
  --file FILE                    Path to summary JSON file (default: summary.json)
  --output DIR                   Output directory for results (default: results)
  --mock                         Use mock LLMs for testing (no API keys)
  --llm-provider {openai, anthropic, llama}
                                 LLM provider to use (default: llama)
  --use-llm-judge               Use LLM3 as judge instead of TF-IDF
  --help                         Show this help message
```

## Examples

### Run Default summaries with Mock LLMs
```bash
python test_with_summaries.py --mock
```

### Run Custom Summary File with LLAMA
```bash
python test_with_summaries.py --file my_summaries.json --llm-provider llama
```

### Run with LLM Judge (Using LLM3 for Similarity)
```bash
python test_with_summaries.py --mock --use-llm-judge
```

### Run with OpenAI instead of LLAMA
```bash
export OPENAI_API_KEY="your_key"
python test_with_summaries.py --llm-provider openai
```

## Summary JSON Format

The script expects a JSON file with one of these formats:

### Format 1: Object with summaries array (Recommended)
```json
{
  "summaries": [
    {"name": "test1", "content": "Summary text here"},
    {"name": "test2", "content": "Another summary"}
  ]
}
```

### Format 2: Simple array of strings
```json
[
  "First summary text",
  "Second summary text"
]
```

## Output

The script generates results in the `results/` directory:

### Results File: `results/pipeline_results.json`
```json
{
  "execution_time": "2026-03-10T10:30:45.123456",
  "total_tests": 3,
  "passed": 2,
  "failed": 1,
  "results": [
    {
      "summary_name": "authentication_system",
      "success": true,
      "final_similarity_score": 0.92,
      "total_iterations": 2,
      "timestamp": "2026-03-10T10:30:45.123456"
    },
    ...
  ]
}
```

## Understanding the Output

Console Output:
```
════════════════════════════════════════════════════════════════════════════════
Processing: authentication_system
════════════════════════════════════════════════════════════════════════════════
Input summary: The authentication system must verify user credentials ...

...pipeline execution details...

════════════════════════════════════════════════════════════════════════════════
PIPELINE EXECUTION SUMMARY
════════════════════════════════════════════════════════════════════════════════
Status: SUCCESS ✓
Total Iterations: 2/5
Final Similarity Score: 0.9247
Threshold: 0.85
...

════════════════════════════════════════════════════════════════════════════════
FINAL SUMMARY
════════════════════════════════════════════════════════════════════════════════

Total Tests: 3
✓ Passed: 2
✗ Failed: 1
Average Similarity Score: 0.9105

Detailed Results:
  1. authentication_system: ✓ PASS (Score: 0.9247, Iterations: 2)
  2. data_processing: ✓ PASS (Score: 0.8956, Iterations: 3)
  3. api_specification: ✗ FAIL (Error: API rate limit exceeded...)

════════════════════════════════════════════════════════════════════════════════
```

## Tips

1. **Start with Mock LLMs** - Test your summaries without API costs
2. **Use TF-IDF by default** - Faster and cheaper than LLM judge
3. **Adjust Threshold** - Modify in `.env` if you're getting too many failures
4. **Use Larger Models for LLM1** - Better specs with 70b model
5. **Use Smaller Models for LLM3** - Just as good for judging but faster

## Troubleshooting

### "FileNotFoundError: summary.json not found"
Create `summary.json` first:
```bash
cp summary.json.example summary.json
# Edit summary.json with your summaries
```

Or run without the file - script will create an example:
```bash
python test_with_summaries.py --mock
```

### "API key not found"
Set environment variable before running:
```bash
export LLAMA_API_KEY="your_key"
python test_with_summaries.py
```

### Results not saved
Check output directory permissions:
```bash
mkdir -p results
chmod 755 results
```

### Low similarity scores
- Lower the threshold in `.env`
- Use more detailed summaries
- Use larger models (70b instead of 7b)
- Increase max_iterations

## Customization

### Custom Configuration

Edit `config.py` to modify:
- Model names and parameters
- Prompt templates
- Similarity thresholds
- Max iterations

### Add More Summaries

Simply add to `summary.json`:
```json
{
  "summaries": [
    ...existing summaries...,
    {
      "name": "new_feature",
      "content": "Your new summary here"
    }
  ]
}
```

### Batch Processing

Create a loop script to process multiple summary files:
```python
import os
from pathlib import Path

for summary_file in Path(".").glob("summaries_*.json"):
    os.system(f"python test_with_summaries.py --file {summary_file}")
```

## Performance Metrics

Typical execution time (per summary):
- **Mock LLMs**: < 1 second
- **LLAMA 7b**: 5-10 seconds
- **LLAMA 13b**: 10-15 seconds
- **LLAMA 70b**: 15-30 seconds

Typical costs per summary (with LLAMA via Together AI):
- LLM1 (70b): ~$0.005-$0.010
- LLM2 (13b): ~$0.001-$0.002
- LLM3 (7b): ~$0.0005-$0.001
- **Total per summary**: ~$0.007-$0.013
