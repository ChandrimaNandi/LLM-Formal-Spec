#!/usr/bin/env python3
"""
Test script to run the Formal Spec Generation Pipeline using summaries from summary.json
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from config import create_default_config
from pipeline import FormalSpecGenerationPipeline, PipelineResult


class PipelineTester:
    """Test runner for the pipeline using summary.json"""
    
    def __init__(self, summary_file: str = "summary.json", output_dir: str = "results"):
        """
        Initialize the tester
        
        Args:
            summary_file: Path to JSON file containing summaries
            output_dir: Directory to save results
        """
        self.summary_file = summary_file
        self.output_dir = output_dir
        self.results = []
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
    
    def load_summaries(self) -> List[Dict]:
        """
        Load summaries from JSON file
        
        Expected formats:
        1. {"summaries": [{"name": "test1", "content": "Summary text"}]}
        2. {"specifications": [{"id": 1, "title": "Title", "description": "Text"}]}
        3. Simple array: ["Summary 1", "Summary 2"]
        """
        if not os.path.exists(self.summary_file):
            print(f"⚠️  File not found: {self.summary_file}")
            print("Creating example summary.json...\n")
            self._create_example_summary_file()
        
        with open(self.summary_file, "r") as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, dict):
            if "summaries" in data:
                return data["summaries"]
            elif "specifications" in data:
                # Convert specifications format to summaries format
                return [
                    {
                        "name": spec.get('title', f'spec_{spec.get("id", i)}'),
                        "content": spec.get("description", "")
                    }
                    for i, spec in enumerate(data["specifications"])
                ]
            else:
                raise ValueError(f"Unknown JSON format. Expected 'summaries' or 'specifications' key")
        elif isinstance(data, list):
            # Convert simple list to dict format
            return [
                {"name": f"summary_{i}", "content": item}
                for i, item in enumerate(data)
            ]
        else:
            raise ValueError(f"Unexpected JSON format in {self.summary_file}")
    
    def _create_example_summary_file(self):
        """Create an example summary.json file"""
        example_data = {
            "summaries": [
                {
                    "name": "authentication_system",
                    "content": """
                    The authentication system must verify user credentials within 100ms.
                    Failed login attempts are logged and tracked.
                    After 5 failed attempts, the account is temporarily locked for 30 minutes.
                    Session tokens expire after 24 hours of inactivity.
                    All password changes require email verification.
                    """
                },
                {
                    "name": "data_processing",
                    "content": """
                    The system processes user requests and returns responses within 2 seconds.
                    All data must be encrypted at rest (AES-256) and in transit (TLS 1.3).
                    The system must maintain 99.9% uptime.
                    User authentication is required for all operations.
                    Data retention policy: keep data for 7 years then archive.
                    """
                },
                {
                    "name": "api_specification",
                    "content": """
                    The API must support RESTful operations (GET, POST, PUT, DELETE).
                    Rate limiting: 1000 requests per minute per user.
                    Response time must be < 500ms for 95% of requests.
                    All endpoints require OAuth 2.0 authentication.
                    API versioning through URL path (/v1/, /v2/).
                    """
                }
            ]
        }
        
        with open(self.summary_file, "w") as f:
            json.dump(example_data, f, indent=2)
        
        print(f"Created example {self.summary_file}")
    
    def run_pipeline_on_summary(
        self,
        summary: Dict,
        config,
        use_llm_judge: bool = False,
        llm_provider: str = "llama"
    ) -> Dict:
        """
        Run pipeline on a single summary
        
        Args:
            summary: Summary dict with 'name', 'content', and optional 'function' keys
            config: Pipeline configuration
            use_llm_judge: Whether to use LLM3 as judge
            llm_provider: LLM provider to use
            
        Returns:
            Result dictionary
        """
        summary_name = summary.get("name", "unknown")
        summary_text = summary.get("content", "")
        
        # Include function details if available
        function_details = summary.get("function")
        if function_details and not summary_text:
            # If no content but has function, build content from function
            func_name = function_details.get("name", "")
            func_params = function_details.get("parameters", {})
            summary_text = f"Function: {func_name}\nParameters: {json.dumps(func_params, indent=2)}"
        
        print(f"\n{'=' * 80}")
        print(f"Processing: {summary_name}")
        print(f"{'=' * 80}")
        print(f"Input summary: {summary_text[:100]}...")
        
        if function_details:
            print(f"Function: {function_details.get('name', 'N/A')}")
            print(f"Parameters: {function_details.get('parameters', {})}")
        
        print()
        
        try:
            # Create pipeline
            pipeline = FormalSpecGenerationPipeline(
                config,
                use_llm_judge=use_llm_judge,
                similarity_method="tfidf",
                llm_provider=llm_provider
            )
            
            # Run pipeline
            result = pipeline.run(summary_text.strip())
            
            # Print summary
            pipeline.print_summary(result)
            
            # Create result entry
            result_entry = {
                "summary_name": summary_name,
                "success": result.success,
                "final_similarity_score": result.final_similarity_score,
                "total_iterations": result.total_iterations,
                "ground_truth_summary": result.ground_truth_summary[:100],
                "final_summary": result.final_summary[:100],
                "final_formal_spec": result.final_formal_spec[:300],
                "function_name": function_details.get("name") if function_details else None,
                "timestamp": result.timestamp
            }
            
            return result_entry
        
        except Exception as e:
            print(f"❌ Error processing {summary_name}: {str(e)}")
            return {
                "summary_name": summary_name,
                "success": False,
                "error": str(e),
                "function_name": function_details.get("name") if function_details else None,
                "timestamp": datetime.now().isoformat()
            }
    
    def run_all(
        self,
        use_llm_judge: bool = False,
        llm_provider: str = "llama",
        use_mock: bool = False
    ) -> None:
        """
        Run pipeline on all summaries in summary.json
        
        Args:
            use_llm_judge: Whether to use LLM3 as judge
            llm_provider: LLM provider to use
            use_mock: Whether to use mock LLMs for testing
        """
        print(f"\n{'#' * 80}")
        print("# Formal Spec Generation Pipeline - Test Runner")
        print(f"{'#' * 80}\n")
        
        # Load summaries
        summaries = self.load_summaries()
        print(f"Loaded {len(summaries)} summary(ies) from {self.summary_file}\n")
        
        # Create configuration
        if use_mock:
            print("Using MOCK LLMs for testing (no API keys needed)")
            from config import PipelineConfig, LLMConfig
            from llm_client import MockLLMClient
            
            config = PipelineConfig(
                llm1_config=LLMConfig(model_name="mock-1", api_key="mock-key"),
                llm2_config=LLMConfig(model_name="mock-2", api_key="mock-key"),
                llm3_config=LLMConfig(model_name="mock-3", api_key="mock-key"),
                similarity_threshold=0.85,
                max_iterations=2,
                verbose=True
            )
        else:
            print(f"Using {llm_provider.upper()} LLMs for testing")
            config = create_default_config()
        
        # Process each summary
        for summary in summaries:
            result = self.run_pipeline_on_summary(
                summary,
                config,
                use_llm_judge=use_llm_judge,
                llm_provider=llm_provider
            )
            self.results.append(result)
        
        # Save results
        self.save_results()
        
        # Print final summary
        self.print_final_summary()
    
    def save_results(self) -> None:
        """Save results to JSON file"""
        output_file = os.path.join(self.output_dir, "pipeline_results.json")
        
        result_data = {
            "execution_time": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r.get("success", False)),
            "failed": sum(1 for r in self.results if not r.get("success", False)),
            "results": self.results
        }
        
        with open(output_file, "w") as f:
            json.dump(result_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
    
    def print_final_summary(self) -> None:
        """Print final summary of all tests"""
        print(f"\n{'=' * 80}")
        print("FINAL SUMMARY")
        print(f"{'=' * 80}\n")
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get("success", False))
        failed = total - passed
        
        print(f"Total Tests: {total}")
        print(f"✓ Passed: {passed}")
        print(f"✗ Failed: {failed}")
        
        if total > 0:
            avg_similarity = sum(
                r.get("final_similarity_score", 0) 
                for r in self.results 
                if r.get("success", False)
            ) / max(1, passed)
            print(f"Average Similarity Score: {avg_similarity:.4f}")
        
        print(f"\nDetailed Results:")
        for i, result in enumerate(self.results, 1):
            status = "✓ PASS" if result.get("success") else "✗ FAIL"
            name = result.get("summary_name", "unknown")
            if result.get("success"):
                score = result.get("final_similarity_score", 0)
                iters = result.get("total_iterations", 0)
                print(f"  {i}. {name}: {status} (Score: {score:.4f}, Iterations: {iters})")
            else:
                error = result.get("error", "Unknown error")
                print(f"  {i}. {name}: {status} (Error: {error[:50]}...)")
        
        print(f"\n{'=' * 80}\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Formal Spec Generation Pipeline with summary.json"
    )
    parser.add_argument(
        "--file",
        default="summary.json",
        help="Path to JSON file with summaries (default: summary.json)"
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLMs for testing (no API keys needed)"
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "anthropic", "llama"],
        default="llama",
        help="LLM provider to use (default: llama)"
    )
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Use LLM3 as judge instead of TF-IDF"
    )
    
    args = parser.parse_args()
    
    # Create and run tester
    tester = PipelineTester(summary_file=args.file, output_dir=args.output)
    
    try:
        tester.run_all(
            use_llm_judge=args.use_llm_judge,
            llm_provider=args.llm_provider,
            use_mock=args.mock
        )
        return 0
    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
