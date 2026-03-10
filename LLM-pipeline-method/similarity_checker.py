"""
Similarity checker for comparing summaries
"""
from typing import Tuple
from abc import ABC, abstractmethod
from llm_client import LLMClient
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityChecker(ABC):
    """Abstract base class for similarity checking"""
    
    @abstractmethod
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (0-1)"""
        pass


class LLMJudgeSimilarity(SimilarityChecker):
    """Use an LLM as a judge to calculate similarity"""
    
    def __init__(self, judge_llm: LLMClient, judge_prompt_template: str):
        self.judge_llm = judge_llm
        self.judge_prompt_template = judge_prompt_template
    
    def calculate_similarity(self, original_summary: str, generated_summary: str) -> float:
        """Use LLM to judge similarity"""
        prompt = self.judge_prompt_template.format(
            original_summary=original_summary,
            generated_summary=generated_summary
        )
        
        try:
            similarity_score = self.judge_llm.generate_number(prompt)
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, similarity_score))
        except Exception as e:
            raise RuntimeError(f"Error calculating similarity with LLM: {str(e)}")


class TFIDFSimilarity(SimilarityChecker):
    """Use TF-IDF and cosine similarity"""
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF cosine similarity"""
        if not text1 or not text2:
            return 0.0
        
        try:
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            return float(similarity)
        except Exception as e:
            raise RuntimeError(f"Error calculating TF-IDF similarity: {str(e)}")


class SemanticSimilarity(SimilarityChecker):
    """Use semantic similarity (requires embedding model)"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(embedding_model)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Install with: "
                "pip install sentence-transformers"
            )
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embeddings"""
        if not text1 or not text2:
            return 0.0
        
        try:
            embedding1 = self.model.encode(text1, convert_to_tensor=True)
            embedding2 = self.model.encode(text2, convert_to_tensor=True)
            
            # Calculate cosine similarity
            from torch.nn.functional import cosine_similarity
            similarity = cosine_similarity(embedding1, embedding2, dim=0)
            return float(similarity)
        except Exception as e:
            raise RuntimeError(f"Error calculating semantic similarity: {str(e)}")


class HybridSimilarity(SimilarityChecker):
    """Combine multiple similarity metrics"""
    
    def __init__(self, checkers: list, weights: list):
        """
        Args:
            checkers: List of SimilarityChecker instances
            weights: List of weights for each checker (should sum to 1)
        """
        if len(checkers) != len(weights):
            raise ValueError("Number of checkers must match number of weights")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1")
        
        self.checkers = checkers
        self.weights = weights
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate weighted average of similarities"""
        total_similarity = 0.0
        for checker, weight in zip(self.checkers, self.weights):
            similarity = checker.calculate_similarity(text1, text2)
            total_similarity += weight * similarity
        
        return total_similarity


def create_similarity_checker(
    method: str = "tfidf",
    judge_llm: LLMClient = None,
    judge_prompt_template: str = None
) -> SimilarityChecker:
    """Factory function to create similarity checker"""
    if method.lower() == "llm":
        if judge_llm is None:
            raise ValueError("judge_llm required for LLM method")
        if judge_prompt_template is None:
            raise ValueError("judge_prompt_template required for LLM method")
        return LLMJudgeSimilarity(judge_llm, judge_prompt_template)
    elif method.lower() == "tfidf":
        return TFIDFSimilarity()
    elif method.lower() == "semantic":
        return SemanticSimilarity()
    else:
        raise ValueError(f"Unknown similarity method: {method}")
