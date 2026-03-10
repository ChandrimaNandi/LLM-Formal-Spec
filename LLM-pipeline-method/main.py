"""
Main entry point for the Formal Spec Generation Pipeline
"""
import argparse
import sys
from config import create_default_config
from pipeline import FormalSpecGenerationPipeline


def example_usage_mock():
    """Example usage with mock LLMs for testing"""
    print("Running example with MOCK LLMs (no API keys needed)...")
    
    from config import PipelineConfig, LLMConfig
    from llm_client import MockLLMClient
    
    # Create mock config
    config = PipelineConfig(
        llm1_config=LLMConfig(
            model_name="mock-1",
            api_key="mock-key-1"
        ),
        llm2_config=LLMConfig(
            model_name="mock-2",
            api_key="mock-key-2"
        ),
        llm3_config=LLMConfig(
            model_name="mock-3",
            api_key="mock-key-3"
        ),
        similarity_threshold=0.85,
        max_iterations=3,
        verbose=True
    )
    
    # Create pipeline with mock clients
    pipeline = FormalSpecGenerationPipeline(config, use_llm_judge=False, similarity_method="tfidf")
    
    # Override with mock clients
    pipeline.llm1 = MockLLMClient(config.llm1_config)
    pipeline.llm2 = MockLLMClient(config.llm2_config)
    pipeline.llm3 = MockLLMClient(config.llm3_config)
    
    # Example ground truth summary
    ground_truth_summary = """
    The system processes user requests and returns responses within 2 seconds.
    All data must be encrypted at rest and in transit.
    The system must maintain 99.9% uptime.
    User authentication is required for all operations.
    """
    
    # Run pipeline
    result = pipeline.run(ground_truth_summary)
    
    # Print and save results
    pipeline.print_summary(result)
    pipeline.save_results(result, output_file="mock_results.json")
    
    return result


def example_usage_real():
    """Example usage with real LLMs (requires API keys)"""
    print("Running example with REAL LLMs using LLAMA...\n")
    print("Make sure LLAMA_API_KEY environment variable is set\n")
    
    config = create_default_config()
    
    # Create pipeline with LLAMA provider
    pipeline = FormalSpecGenerationPipeline(
        config,
        use_llm_judge=True,  # Use LLM3 as judge
        similarity_method="llm",
        llm_provider="llama"  # Use LLAMA as provider
    )
    
    # Example ground truth summary
    ground_truth_summary = """
    The authentication system must verify user credentials within 100ms.
    Failed login attempts are logged and tracked.
    After 5 failed attempts, the account is temporarily locked for 30 minutes.
    Session tokens expire after 24 hours of inactivity.
    All password changes require email verification.
    """
    
    # Run pipeline
    result = pipeline.run(ground_truth_summary)
    
    # Print and save results
    pipeline.print_summary(result)
    pipeline.save_results(result, output_file="real_results.json")
    
    return result


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Formal Specification Generation Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "real"],
        default="mock",
        help="Run with mock LLMs or real API calls (default: mock)"
    )
    parser.add_argument(
        "--summary",
        type=str,
        help="Ground truth summary to process"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Similarity threshold (0-1)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Maximum iterations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pipeline_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--similarity-method",
        choices=["tfidf", "semantic", "llm"],
        default="tfidf",
        help="Similarity checking method"
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "anthropic", "llama"],
        default="llama",
        help="LLM provider to use (default: llama)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "mock":
            result = example_usage_mock()
        else:
            result = example_usage_real()
        
        return 0 if result.success else 1
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
