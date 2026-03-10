"""
LLM Client for interacting with various LLM providers
"""
from abc import ABC, abstractmethod
from typing import Optional
from config import LLMConfig


class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt"""
        pass
    
    @abstractmethod
    def generate_number(self, prompt: str) -> float:
        """Generate a number from a prompt (for judge LLM)"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI LLM Client"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import openai
            openai.api_key = config.api_key
            self.client = openai.OpenAI(api_key=config.api_key)
        except ImportError:
            raise ImportError("openai package not installed. Install it with: pip install openai")
    
    def generate(self, prompt: str) -> str:
        """Generate text using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Error calling OpenAI API: {str(e)}")
    
    def generate_number(self, prompt: str) -> float:
        """Generate a number using OpenAI API"""
        response = self.generate(prompt)
        try:
            return float(response)
        except ValueError:
            raise ValueError(f"Expected a float, got: {response}")


class AnthropicClient(LLMClient):
    """Anthropic Claude LLM Client"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=config.api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Install it with: pip install anthropic")
    
    def generate(self, prompt: str) -> str:
        """Generate text using Anthropic API"""
        try:
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            return response.content[0].text.strip()
        except Exception as e:
            raise RuntimeError(f"Error calling Anthropic API: {str(e)}")
    
    def generate_number(self, prompt: str) -> float:
        """Generate a number using Anthropic API"""
        response = self.generate(prompt)
        try:
            return float(response)
        except ValueError:
            raise ValueError(f"Expected a float, got: {response}")


class LlamaClient(LLMClient):
    """LLAMA LLM Client (using Together AI or similar service)"""
    
    def __init__(self, config: LLMConfig, base_url: str = "https://api.together.xyz/v1"):
        super().__init__(config)
        self.base_url = base_url
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=config.api_key,
                base_url=base_url
            )
        except ImportError:
            raise ImportError("openai package required for LLAMA client. Install with: pip install openai")
    
    def generate(self, prompt: str) -> str:
        """Generate text using LLAMA API"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Error calling LLAMA API: {str(e)}")
    
    def generate_number(self, prompt: str) -> float:
        """Generate a number using LLAMA API"""
        response = self.generate(prompt)
        try:
            # Handle potential whitespace or formatting issues
            return float(response.strip())
        except ValueError:
            raise ValueError(f"Expected a float, got: {response}")


class MockLLMClient(LLMClient):
    """Mock LLM Client for testing"""
    
    def __init__(self, config: LLMConfig, response_map: Optional[dict] = None):
        super().__init__(config)
        self.response_map = response_map or {}
        self.call_count = 0
    
    def generate(self, prompt: str) -> str:
        """Return mock response"""
        self.call_count += 1
        
        # Check if there's a response for this prompt
        for key, value in self.response_map.items():
            if key.lower() in prompt.lower():
                return value
        
        # Default mock responses
        if "formal" in prompt.lower():
            return f"Mock Formal Specification {self.call_count}\n\nProperty 1: ∀x ∈ System: (precondition → postcondition)\nProperty 2: Invariant(state)"
        elif "summary" in prompt.lower():
            return f"Mock Summary {self.call_count}: This is a generated summary based on formal specifications."
        else:
            return "Mock Response"
    
    def generate_number(self, prompt: str) -> float:
        """Return mock similarity score"""
        self.call_count += 1
        return 0.92  # Mock high similarity


def create_llm_client(config: LLMConfig, provider: str = "openai", base_url: str = None) -> LLMClient:
    """Factory function to create LLM client based on provider"""
    if provider.lower() == "openai":
        return OpenAIClient(config)
    elif provider.lower() == "anthropic":
        return AnthropicClient(config)
    elif provider.lower() == "llama":
        if base_url:
            return LlamaClient(config, base_url=base_url)
        else:
            return LlamaClient(config)
    elif provider.lower() == "mock":
        return MockLLMClient(config)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
