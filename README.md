# LLM Formal Specification Pipeline

## Overview

This repository contains multiple methods for generating and verifying formal specifications from natural language descriptions.

## LLM-sts-pipeline

Architecture

- **LLM 1** — Formal Spec Generator: Uses chain-of-thought reasoning with 3-step pipeline (Annotation → Lifting → Translation)
- **LLM 2** — Summariser: Converts formal specifications into human-readable draft summaries
- **BERTScore Judge** — Replaces traditional LLM-based judging with sophisticated semantic and logical consistency scoring:
  - Uses `cross-encoder/stsb-roberta-large` for semantic similarity (STS) scoring
  - Uses `cross-encoder/nli-deberta-v3-base` for logical/negation consistency (NLI) scoring
- **Intelligent Refinement Loop** — Failed specifications automatically feed back into LLM 1 for refinement until quality threshold is reached

---

## Prerequisites

### System Requirements

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended for batch processing)
- GPU support is optional (CPU works fine with sentence-transformers)

### API Keys

- **LLM Provider API Key** — Required for accessing the language models used in the pipeline
  - The pipeline uses various LLM models for specification generation and summarization

### Input Files

You need a `summary.json` file with the ground-truth data you want to generate formal specifications for.

---

## Installation & Setup

### Step 1: Clone or Download the Repository

```bash
cd /path/to/LLM-Formal-Spec-main
```

### Step 2: Install Dependencies

```bash
pip install sentence-transformers
pip install bert-score
pip install pandas
```

Or install everything at once:

```bash
pip install sentence-transformers bert-score pandas
```

### Step 3: Set Environment Variables

Create a `.env` file in the `LLM-sts-pipeline` directory (or set your shell variables) with your LLM provider credentials:

```bash
export LLM_API_KEY="your-api-key-here"
```

Or set it in Python before running:

```python
import os
os.environ['LLM_API_KEY'] = 'your-api-key-here'
```

Refer to the notebook's configuration section for specific provider setup instructions.

### Step 4: Prepare Your Input Data

You should have a `summary.json` file with the ground-truth data. The pipeline will generate formal specifications from this input.

---

## Running the Pipeline

### Option 1: Run as Jupyter Notebook (Recommended for First-Time)

1. Open Jupyter Notebook:

   ```bash
   cd LLM-sts-pipeline
   jupyter notebook formal-spec-sts-llm3.ipynb
   ```
2. Configure the pipeline settings in the **Configuration** cell:

   ```python
   # Adjust these parameters as needed:
   SIMILARITY_THRESHOLD = 0.9      # Quality threshold (0.0-1.0)
   MAX_ITERATIONS = 5              # Max refinement rounds
   BATCH_SIZE = 3                  # LLM requests per batch
   INTER_BATCH_SLEEP = 8           # Seconds between batches
   DATASET_FOLDER = "/path/to/your/data"  # Path to your input data
   ```
3. Run cells in order:

   - Cell 1: Install dependencies
   - Cell 2: Import libraries and configure
   - Cell 3+: Execute the pipeline

### Option 2: Run Programmatically (Python Script)

Create a Python script `run_pipeline.py`:

```python
import os
import sys
sys.path.append('./LLM-sts-pipeline')

# Set your LLM provider API key
os.environ['LLM_API_KEY'] = 'your-api-key'

# Import and run the pipeline code
# (See the notebook for the full implementation)
```

---

## Configuration Options

### Pipeline Settings

| Parameter                | Default   | Description                           |
| ------------------------ | --------- | ------------------------------------- |
| `SIMILARITY_THRESHOLD` | 0.9       | BERTScore threshold to pass (0.0-1.0) |
| `MAX_ITERATIONS`       | 5         | Maximum refinement loops              |
| `BATCH_SIZE`           | 3         | LLM requests processed together       |
| `INTER_BATCH_SLEEP`    | 8 seconds | Wait time between batches             |

### Model Settings

| Parameter      | Default                               | Description                          |
| -------------- | ------------------------------------- | ------------------------------------ |
| `STS_MODEL`  | `cross-encoder/stsb-roberta-large`  | Semantic similarity scorer           |
| `NLI_MODEL`  | `cross-encoder/nli-deberta-v3-base` | Logical consistency scorer           |
| `STS_WEIGHT` | 0.5                                   | Weight for semantic similarity score |
| `NLI_WEIGHT` | 0.5                                   | Weight for logical consistency score |

### LLM Settings

| Parameter       | Default                               | Description         |
| --------------- | ------------------------------------- | ------------------- |
| `MODEL_CHAIN` | `['llama-3.3-70b', 'llama-3.1-8b']` | Fallback LLM models |

---

## Output Files

The pipeline generates the following output files in the working directory:

| File                                 | Description                                  |
| ------------------------------------ | -------------------------------------------- |
| `formal_specifications.json`       | Generated formal specifications              |
| `formal_specifications_iter2.json` | Refined specifications after iteration 2     |
| `draft_summary.json`               | Human-readable summaries of specs            |
| `judge_report.json`                | Detailed scoring report from BERTScore Judge |
| `iteration_log.json`               | Log of all refinement iterations             |

---

## Other Methods (For Reference)

This repository also contains alternative approaches:

- **Lifting method**: Manual lifting-based formal specification generation
- **LLM-pipeline-method**: Two-stage LLM-based generation with BERTScore
- **Self loop verification**: Iterative self-verification of formal specs

While these methods have merit, **LLM-sts-pipeline is recommended** for best results.
