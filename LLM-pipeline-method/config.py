"""
Configuration for the Formal Spec Generation Pipeline
"""
from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for LLM models"""
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60


@dataclass
class PipelineConfig:
    """Configuration for the pipeline"""
    # LLM configurations
    llm1_config: LLMConfig  # Generates formal specs from summary
    llm2_config: LLMConfig  # Generates summary from formal specs
    llm3_config: LLMConfig  # Judges similarity between summaries
    
    # Pipeline parameters
    similarity_threshold: float = 0.85  # Threshold for summary similarity (0-1)
    max_iterations: int = 5  # Maximum number of iterations in the loop
    save_iterations: bool = True  # Whether to save iteration results
    verbose: bool = True  # Verbose logging
    
    # Prompts
    formal_spec_prompt_template: str = """
You are a formal specification expert. Convert the following functional specification into a formal specification using mathematical notation, logical operators, and/or formal methods notation (Z notation, TLA+, or similar).

Include:
1. Pre-conditions and post-conditions
2. Invariants that must hold
3. State transitions if applicable
4. Formal constraints and rules

Specification:
{summary}

Generate a detailed and rigorous formal specification:
"""
    
    summary_from_spec_prompt_template: str = """
You are a specification analyst. Read the following formal specification and generate a concise natural language summary of its key requirements and properties.

Formal Specification:
{formal_spec}

Generate a summary:
"""
    
    judge_prompt_template: str = """
You are a semantic similarity expert. Compare the following two summaries and rate their semantic similarity on a scale from 0 to 1, where:
- 0 means completely different
- 0.5 means partially similar
- 1 means identical/nearly identical

Original Summary:
{original_summary}

Generated Summary:
{generated_summary}

Respond with ONLY a float between 0 and 1. No other text.
"""


def create_default_config() -> PipelineConfig:
    """Create a default configuration from environment variables"""
    # Support both LLAMA and OpenAI API keys
    llm_api_key = os.getenv("LLAMA_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    
    llm1_config = LLMConfig(
        model_name=os.getenv("LLM1_MODEL", "meta-llama/Llama-2-70b-chat-hf"),
        api_key=llm_api_key,
        temperature=float(os.getenv("LLM1_TEMPERATURE", "0.7")),
    )
    
    llm2_config = LLMConfig(
        model_name=os.getenv("LLM2_MODEL", "meta-llama/Llama-2-13b-chat-hf"),
        api_key=llm_api_key,
        temperature=float(os.getenv("LLM2_TEMPERATURE", "0.5")),
    )
    
    llm3_config = LLMConfig(
        model_name=os.getenv("LLM3_MODEL", "meta-llama/Llama-2-7b-chat-hf"),
        api_key=llm_api_key,
        temperature=float(os.getenv("LLM3_TEMPERATURE", "0.0")),
    )
    
    return PipelineConfig(
        llm1_config=llm1_config,
        llm2_config=llm2_config,
        llm3_config=llm3_config,
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.85")),
        max_iterations=int(os.getenv("MAX_ITERATIONS", "5")),
        verbose=os.getenv("VERBOSE", "true").lower() == "true",
    )
