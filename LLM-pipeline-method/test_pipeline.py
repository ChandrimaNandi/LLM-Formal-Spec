#!/usr/bin/env python3
"""
Quick test script for the Formal Spec Generation Pipeline with LLAMA
"""
import os
import sys

def test_mock_llm():
    """Test with mock LLMs (no API keys needed)"""
    print("=" * 80)
    print("TEST 1: Mock LLMs (No API Keys)")
    print("=" * 80)
    
    from config import PipelineConfig, LLMConfig
    from llm_client import MockLLMClient
    from pipeline import FormalSpecGenerationPipeline
    
    config = PipelineConfig(
        llm1_config=LLMConfig(model_name="mock-1", api_key="mock-key"),
        llm2_config=LLMConfig(model_name="mock-2", api_key="mock-key"),
        llm3_config=LLMConfig(model_name="mock-3", api_key="mock-key"),
        similarity_threshold=0.85,
        max_iterations=2,
        verbose=True
    )
    
    pipeline = FormalSpecGenerationPipeline(config, use_llm_judge=False, similarity_method="tfidf")
    
    # Override with mock clients
    pipeline.llm1 = MockLLMClient(config.llm1_config)
    pipeline.llm2 = MockLLMClient(config.llm2_config)
    pipeline.llm3 = MockLLMClient(config.llm3_config)
    
    summary = "The system must ensure data consistency and handle failures gracefully."
    result = pipeline.run(summary)
    pipeline.print_summary(result)
    
    return result.success


def test_llama_connection():
    """Test LLAMA API connection"""
    print("\n" + "=" * 80)
    print("TEST 2: LLAMA API Connection")
    print("=" * 80)
    
    llama_key = os.getenv("LLAMA_API_KEY")
    if not llama_key:
        print("❌ LLAMA_API_KEY not found in environment")
        print("   Set it with: export LLAMA_API_KEY='your_key_here'")
        return False
    
    print(f"✓ LLAMA_API_KEY found (length: {len(llama_key)})")
    
    try:
        from config import LLMConfig
        from llm_client import LlamaClient
        
        config = LLMConfig(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            api_key=llama_key,
            temperature=0.0,
            max_tokens=100
        )
        
        client = LlamaClient(config)
        print("✓ LLAMA client created successfully")
        
        # Try a simple test
        response = client.generate("What is 2+2? Answer with just the number.")
        print(f"✓ LLAMA response: {response[:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error testing LLAMA connection: {str(e)}")
        return False


def test_full_pipeline_llama():
    """Test full pipeline with LLAMA"""
    print("\n" + "=" * 80)
    print("TEST 3: Full Pipeline with LLAMA")
    print("=" * 80)
    
    llama_key = os.getenv("LLAMA_API_KEY")
    if not llama_key:
        print("❌ LLAMA_API_KEY not found. Skipping full pipeline test.")
        return False
    
    try:
        from config import create_default_config
        from pipeline import FormalSpecGenerationPipeline
        
        config = create_default_config()
        pipeline = FormalSpecGenerationPipeline(
            config,
            use_llm_judge=False,  # Use TF-IDF to avoid extra LLM calls
            similarity_method="tfidf",
            llm_provider="llama"
        )
        
        summary = "The system requires user authentication and authorization checks."
        result = pipeline.run(summary)
        pipeline.print_summary(result)
        
        return result.success
    except Exception as e:
        print(f"❌ Error in full pipeline: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("Formal Spec Generation Pipeline - Test Suite")
    print("=" * 80 + "\n")
    
    results = {}
    
    # Test 1: Mock LLMs
    try:
        results["mock_llm"] = test_mock_llm()
    except Exception as e:
        print(f"Test 1 Error: {e}")
        results["mock_llm"] = False
    
    # Test 2: LLAMA Connection
    try:
        results["llama_connection"] = test_llama_connection()
    except Exception as e:
        print(f"Test 2 Error: {e}")
        results["llama_connection"] = False
    
    # Test 3: Full Pipeline
    try:
        results["full_pipeline"] = test_full_pipeline_llama()
    except Exception as e:
        print(f"Test 3 Error: {e}")
        results["full_pipeline"] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print("=" * 80 + "\n")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
