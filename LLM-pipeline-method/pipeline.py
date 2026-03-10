"""
Main pipeline orchestrator for formal spec generation
"""
from dataclasses import dataclass, asdict
import json
from typing import Optional, List
from datetime import datetime
from config import PipelineConfig
from llm_client import LLMClient, create_llm_client
from similarity_checker import SimilarityChecker, create_similarity_checker
from logger import PipelineLogger


@dataclass
class IterationResult:
    """Data class for storing iteration results"""
    iteration: int
    ground_truth_summary: str
    formal_spec: str
    generated_summary: str
    similarity_score: float
    passed: bool


@dataclass
class PipelineResult:
    """Data class for storing final pipeline results"""
    success: bool
    ground_truth_summary: str
    final_formal_spec: str
    final_summary: str
    final_similarity_score: float
    total_iterations: int
    iterations: List[IterationResult]
    timestamp: str


class FormalSpecGenerationPipeline:
    """
    Pipeline for automatic formal specification generation with feedback loop
    
    Flow:
    1. Start with ground truth summary
    2. LLM1: Generate formal spec from summary
    3. LLM2: Generate new summary from formal spec
    4. LLM3: Judge similarity between original and new summary
    5. If similarity >= threshold: DONE
    6. If similarity < threshold: Loop back to step 2
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        use_llm_judge: bool = False,
        similarity_method: str = "tfidf",
        llm_provider: str = "llama"
    ):
        """
        Initialize the pipeline
        
        Args:
            config: PipelineConfig instance
            use_llm_judge: Whether to use LLM3 as judge (True) or TF-IDF (False)
            similarity_method: Method for similarity checking ("tfidf", "semantic", "llm")
            llm_provider: LLM provider to use ("openai", "anthropic", "llama")
        """
        self.config = config
        self.logger = PipelineLogger(verbose=config.verbose)
        
        # Create LLM clients
        self.llm1 = create_llm_client(config.llm1_config, provider=llm_provider)
        self.llm2 = create_llm_client(config.llm2_config, provider=llm_provider)
        self.llm3 = create_llm_client(config.llm3_config, provider=llm_provider) if use_llm_judge else None
        
        # Create similarity checker
        if similarity_method.lower() == "llm" and self.llm3 is not None:
            self.similarity_checker = create_similarity_checker(
                method="llm",
                judge_llm=self.llm3,
                judge_prompt_template=config.judge_prompt_template
            )
        else:
            self.similarity_checker = create_similarity_checker(method=similarity_method)
        
        self.iterations: List[IterationResult] = []
    
    def generate_formal_spec(self, summary: str) -> str:
        """
        Use LLM1 to generate formal specification from summary
        
        Args:
            summary: Natural language summary
            
        Returns:
            Formal specification
        """
        self.logger.debug(f"Generating formal spec from summary: {summary[:100]}...")
        
        prompt = self.config.formal_spec_prompt_template.format(summary=summary)
        formal_spec = self.llm1.generate(prompt)
        
        self.logger.debug(f"Generated formal spec ({len(formal_spec)} chars)")
        return formal_spec
    
    def generate_summary_from_spec(self, formal_spec: str) -> str:
        """
        Use LLM2 to generate summary from formal specification
        
        Args:
            formal_spec: Formal specification
            
        Returns:
            Generated summary
        """
        self.logger.debug(f"Generating summary from formal spec...")
        
        prompt = self.config.summary_from_spec_prompt_template.format(
            formal_spec=formal_spec
        )
        summary = self.llm2.generate(prompt)
        
        self.logger.debug(f"Generated summary ({len(summary)} chars)")
        return summary
    
    def check_similarity(
        self,
        original_summary: str,
        generated_summary: str
    ) -> float:
        """
        Use LLM3 or similarity metric to judge summary similarity
        
        Args:
            original_summary: Ground truth summary
            generated_summary: Generated summary
            
        Returns:
            Similarity score (0-1)
        """
        self.logger.debug("Checking similarity between summaries...")
        
        similarity = self.similarity_checker.calculate_similarity(
            original_summary,
            generated_summary
        )
        
        self.logger.debug(f"Similarity score: {similarity:.4f}")
        return similarity
    
    def run_iteration(
        self,
        ground_truth_summary: str,
        iteration_num: int
    ) -> IterationResult:
        """
        Run a single iteration of the pipeline
        
        Args:
            ground_truth_summary: Original summary (ground truth)
            iteration_num: Iteration number
            
        Returns:
            IterationResult instance
        """
        self.logger.info(f"\n--- Iteration {iteration_num} ---")
        
        # Step 1: Generate formal specification
        formal_spec = self.generate_formal_spec(ground_truth_summary)
        
        # Step 2: Generate summary from formal spec
        generated_summary = self.generate_summary_from_spec(formal_spec)
        
        # Step 3: Check similarity
        similarity = self.check_similarity(ground_truth_summary, generated_summary)
        
        # Check if similarity passes threshold
        passed = similarity >= self.config.similarity_threshold
        
        # Log iteration results
        self.logger.log_iteration(
            iteration_num,
            len(formal_spec),
            len(generated_summary),
            similarity
        )
        
        if passed:
            self.logger.info(f"✓ Iteration {iteration_num} PASSED (similarity: {similarity:.4f})")
        else:
            self.logger.info(
                f"✗ Iteration {iteration_num} FAILED "
                f"(similarity: {similarity:.4f} < threshold: {self.config.similarity_threshold})"
            )
        
        # Create result object
        result = IterationResult(
            iteration=iteration_num,
            ground_truth_summary=ground_truth_summary,
            formal_spec=formal_spec,
            generated_summary=generated_summary,
            similarity_score=similarity,
            passed=passed
        )
        
        self.iterations.append(result)
        return result
    
    def run(self, ground_truth_summary: str) -> PipelineResult:
        """
        Run the complete pipeline
        
        Args:
            ground_truth_summary: Original summary (ground truth)
            
        Returns:
            PipelineResult instance
        """
        self.logger.log_pipeline_start(ground_truth_summary)
        
        final_result = None
        iteration = 0
        
        while iteration < self.config.max_iterations:
            iteration += 1
            
            # Run iteration
            result = self.run_iteration(ground_truth_summary, iteration)
            final_result = result
            
            # Check if passed
            if result.passed:
                self.logger.info(
                    f"\n✓ SUCCESS: Formal specifications passed quality check!"
                )
                break
        
        # Log pipeline end
        success = final_result.passed if final_result else False
        final_similarity = final_result.similarity_score if final_result else 0.0
        
        self.logger.log_pipeline_end(final_similarity, iteration, success)
        
        # Create and return pipeline result
        pipeline_result = PipelineResult(
            success=success,
            ground_truth_summary=ground_truth_summary,
            final_formal_spec=final_result.formal_spec if final_result else "",
            final_summary=final_result.generated_summary if final_result else "",
            final_similarity_score=final_similarity,
            total_iterations=iteration,
            iterations=self.iterations,
            timestamp=datetime.now().isoformat()
        )
        
        return pipeline_result
    
    def save_results(
        self,
        result: PipelineResult,
        output_file: str = "pipeline_results.json"
    ) -> None:
        """
        Save pipeline results to JSON file, including all iteration details.
        Each iteration's info is saved in the output file.
        """
        self.logger.info(f"Saving results to {output_file}")

        # Convert to dictionary with all iteration details
        result_dict = {
            "success": result.success,
            "ground_truth_summary": result.ground_truth_summary,
            "final_formal_spec": result.final_formal_spec[:500] + "..." if len(result.final_formal_spec) > 500 else result.final_formal_spec,
            "final_summary": result.final_summary,
            "final_similarity_score": result.final_similarity_score,
            "total_iterations": result.total_iterations,
            "timestamp": result.timestamp,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "ground_truth_summary": it.ground_truth_summary,
                    "formal_spec": it.formal_spec,
                    "generated_summary": it.generated_summary,
                    "similarity_score": it.similarity_score,
                    "passed": it.passed
                }
                for it in result.iterations
            ]
        }

        with open(output_file, "w") as f:
            json.dump(result_dict, f, indent=2)

        self.logger.info(f"Results saved successfully")
    
    def print_summary(self, result: PipelineResult) -> None:
        """Print a summary of the pipeline results"""
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Status: {'SUCCESS ✓' if result.success else 'FAILED ✗'}")
        print(f"Total Iterations: {result.total_iterations}/{self.config.max_iterations}")
        print(f"Final Similarity Score: {result.final_similarity_score:.4f}")
        print(f"Threshold: {self.config.similarity_threshold:.4f}")
        print(f"Timestamp: {result.timestamp}")
        print("\nIteration Details:")
        for it in result.iterations:
            status = "✓ PASS" if it.passed else "✗ FAIL"
            print(f"  Iteration {it.iteration}: Similarity={it.similarity_score:.4f} [{status}]")
        print("=" * 80 + "\n")
