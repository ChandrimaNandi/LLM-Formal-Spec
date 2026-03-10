"""
Logging utility for the pipeline
"""
import logging
from typing import Optional
from datetime import datetime


class PipelineLogger:
    """Logger for the formal spec generation pipeline"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logger = logging.getLogger("FormalSpecPipeline")
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # Add console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        
        # Add handler to logger
        if not self.logger.handlers:
            self.logger.addHandler(ch)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def log_iteration(self, iteration: int, formalspec_length: int, 
                      new_summary_length: int, similarity: float):
        """Log iteration results"""
        self.info(
            f"Iteration {iteration}: Formal Spec Length: {formalspec_length}, "
            f"Summary Length: {new_summary_length}, Similarity: {similarity:.4f}"
        )
    
    def log_pipeline_start(self, ground_truth_summary: str):
        """Log pipeline start"""
        self.info("=" * 80)
        self.info("Starting Formal Spec Generation Pipeline")
        self.info(f"Ground Truth Summary: {ground_truth_summary[:100]}...")
        self.info("=" * 80)
    
    def log_pipeline_end(self, final_similarity: float, iterations: int, success: bool):
        """Log pipeline end"""
        self.info("=" * 80)
        status = "SUCCESS" if success else "FAILED"
        self.info(f"Pipeline {status}")
        self.info(f"Final Similarity: {final_similarity:.4f}")
        self.info(f"Total Iterations: {iterations}")
        self.info("=" * 80)
