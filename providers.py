"""Provider-agnostic model interface for belief probes.

Supports:
- OpenAI: True logprobs via Completions API
- Gemini: True logprobs via Vertex AI
- Ollama: Sampling fallback (no logprobs)

Usage:
    from providers import get_provider

    model = get_provider("openai", model_name="gpt-4o-mini")
    probs = model.get_action_probs(prompt)  # Returns exact softmax over A/B/C/D
"""

import json
import math
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Load .env
def _load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

_load_dotenv()

OPTIONS = ["A", "B", "C", "D"]


@dataclass
class ActionResult:
    """Result from get_action_probs."""
    probs: Dict[str, float]  # Normalized probabilities over A/B/C/D
    entropy: float
    method: str  # "logprobs" or "sampling"
    raw_logprobs: Optional[Dict[str, float]] = None  # Raw log probabilities if available
    n_samples: Optional[int] = None  # Number of samples if sampling method
    metadata: Optional[dict] = None


@dataclass
class ReportResult:
    """Result from get_report."""
    probs: Dict[str, float]
    raw_response: str
    parse_ok: bool
    is_uniform: bool


class BaseProvider(ABC):
    """Base class for model providers."""

    provider_name: str = "base"
    supports_logprobs: bool = False

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def get_action_probs(
        self,
        prompt: str,
        temperature: float = 1.0,
    ) -> ActionResult:
        """Get probability distribution over A/B/C/D at decision point.

        Returns exact logprobs if supported, otherwise samples.
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
    ) -> str:
        """Generate text completion."""
        pass

    def get_report(
        self,
        prompt: str,
        introspective: bool = False,
        anti_uniform: bool = False,
    ) -> ReportResult:
        """Get self-reported probability distribution."""
        constraint = ""
        if anti_uniform:
            constraint = """
CONSTRAINT: You must allocate at least 5% and at most 80% to any option unless logically certain.
Do not use 25/25/25/25 — express your actual confidence distribution."""

        if introspective:
            report_prompt = prompt + f"""

Report the probabilities that governed your choice *immediately before you answered*.
What was your internal confidence for each option at the moment of decision?
{constraint}
Return EXACTLY this JSON format, no other text:
{{"A":25,"B":25,"C":25,"D":25}}"""
        else:
            report_prompt = prompt + f"""

Report your probability for each option as percentages that sum to 100.
{constraint}
Return EXACTLY this JSON format, no other text:
{{"A":25,"B":25,"C":25,"D":25}}"""

        raw_response = self.generate(report_prompt, max_tokens=100)
        probs, parse_ok = self._parse_report(raw_response)

        # Check if uniform
        is_uniform = all(abs(p - 0.25) < 0.02 for p in probs.values())

        return ReportResult(
            probs=probs,
            raw_response=raw_response,
            parse_ok=parse_ok,
            is_uniform=is_uniform,
        )

    def _parse_report(self, text: str) -> Tuple[Dict[str, float], bool]:
        """Parse model's self-reported probabilities."""
        probs = {}
        parse_ok = False

        # Try JSON parsing first
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                for opt in OPTIONS:
                    if opt in data:
                        val = float(data[opt])
                        if val > 1:
                            val = val / 100.0
                        probs[opt] = val
                if len(probs) == 4:
                    parse_ok = True
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        # Fall back to regex
        if not parse_ok:
            probs = {}
            for opt in OPTIONS:
                patterns = [
                    rf"{opt}\s*[:\=]\s*(\d+(?:\.\d+)?)\s*%?",
                    rf"{opt}\)\s*[:\=]?\s*(\d+(?:\.\d+)?)\s*%?",
                ]
                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        probs[opt] = float(match.group(1)) / 100.0
                        break
            parse_ok = len(probs) >= 3

        # Fill missing with uniform
        for opt in OPTIONS:
            if opt not in probs:
                probs[opt] = 0.25

        # Renormalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        return probs, parse_ok

    @staticmethod
    def _entropy(probs: Dict[str, float]) -> float:
        """Shannon entropy in bits."""
        return -sum(p * math.log2(p) for p in probs.values() if p > 0)

    @staticmethod
    def _softmax_from_logprobs(logprobs: Dict[str, float]) -> Dict[str, float]:
        """Convert log probabilities to normalized probabilities."""
        # Subtract max for numerical stability
        max_lp = max(logprobs.values())
        exp_probs = {k: math.exp(v - max_lp) for k, v in logprobs.items()}
        total = sum(exp_probs.values())
        return {k: v / total for k, v in exp_probs.items()}


class OllamaProvider(BaseProvider):
    """Ollama provider using sampling (no logprobs)."""

    provider_name = "ollama"
    supports_logprobs = False

    def __init__(self, model_name: str = "llama3.1:latest", n_samples: int = 50, **kwargs):
        super().__init__(model_name, **kwargs)
        self.n_samples = n_samples
        self.base_url = kwargs.get("base_url", "http://localhost:11434")

        import requests
        self._requests = requests

    def get_action_probs(
        self,
        prompt: str,
        temperature: float = 1.0,
    ) -> ActionResult:
        """Estimate action distribution via sampling."""
        counts = {opt: 0 for opt in OPTIONS}

        for _ in range(self.n_samples):
            response = self._requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 1,
                        "temperature": temperature,
                    }
                }
            )
            text = response.json().get("response", "").strip()

            if text:
                first_char = text[0].upper()
                if first_char in OPTIONS:
                    counts[first_char] += 1

        # Laplace smoothing
        total = sum(counts.values()) + len(OPTIONS)
        probs = {opt: (counts[opt] + 1) / total for opt in OPTIONS}

        return ActionResult(
            probs=probs,
            entropy=self._entropy(probs),
            method="sampling",
            n_samples=self.n_samples,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
    ) -> str:
        response = self._requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                }
            }
        )
        response.raise_for_status()
        return response.json()["response"]


class OpenAIProvider(BaseProvider):
    """OpenAI provider using logprobs."""

    provider_name = "openai"
    supports_logprobs = True

    def __init__(self, model_name: str = "gpt-4o-mini", **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

        import requests
        self._requests = requests

    def get_action_probs(
        self,
        prompt: str,
        temperature: float = 1.0,
    ) -> ActionResult:
        """Get exact action distribution via logprobs."""
        # Use chat completions with logprobs
        response = self._requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "temperature": temperature,
                "logprobs": True,
                "top_logprobs": 20,  # Get top 20 to likely include all of A/B/C/D
            }
        )
        response.raise_for_status()
        data = response.json()

        # Extract logprobs for A, B, C, D
        logprobs = {}
        content = data["choices"][0].get("logprobs", {})
        if content and "content" in content and content["content"]:
            top_logprobs = content["content"][0].get("top_logprobs", [])
            for entry in top_logprobs:
                token = entry["token"].strip().upper()
                if token in OPTIONS:
                    logprobs[token] = entry["logprob"]

        # Fill missing options with very low probability
        for opt in OPTIONS:
            if opt not in logprobs:
                logprobs[opt] = -100.0  # Effectively zero

        probs = self._softmax_from_logprobs(logprobs)

        return ActionResult(
            probs=probs,
            entropy=self._entropy(probs),
            method="logprobs",
            raw_logprobs=logprobs,
            metadata={"model": self.model_name},
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
    ) -> str:
        response = self._requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class GeminiProvider(BaseProvider):
    """Gemini provider using logprobs via Vertex AI or AI Studio."""

    provider_name = "gemini"
    supports_logprobs = True

    def __init__(self, model_name: str = "gemini-2.0-flash", **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")

        import requests
        self._requests = requests

        # AI Studio endpoint
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def get_action_probs(
        self,
        prompt: str,
        temperature: float = 1.0,
    ) -> ActionResult:
        """Get action distribution via Gemini API with logprobs."""
        response = self._requests.post(
            f"{self.base_url}/models/{self.model_name}:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": self.api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": 1,
                    "temperature": temperature,
                    "responseLogprobs": True,
                    "logprobs": 20,  # Top-k logprobs
                },
            }
        )
        response.raise_for_status()
        data = response.json()

        # Extract logprobs
        logprobs = {}
        candidates = data.get("candidates", [])
        if candidates:
            logprobs_result = candidates[0].get("logprobsResult", {})
            top_candidates = logprobs_result.get("topCandidates", [])
            if top_candidates:
                for candidate in top_candidates[0].get("candidates", []):
                    token = candidate.get("token", "").strip().upper()
                    if token in OPTIONS:
                        logprobs[token] = candidate.get("logProbability", -100)

        # Fill missing
        for opt in OPTIONS:
            if opt not in logprobs:
                logprobs[opt] = -100.0

        probs = self._softmax_from_logprobs(logprobs)

        return ActionResult(
            probs=probs,
            entropy=self._entropy(probs),
            method="logprobs",
            raw_logprobs=logprobs,
            metadata={"model": self.model_name},
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
    ) -> str:
        response = self._requests.post(
            f"{self.base_url}/models/{self.model_name}:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": self.api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                },
            }
        )
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "")
        return ""


# Provider registry
PROVIDERS = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
}


def get_provider(provider_name: str, **kwargs) -> BaseProvider:
    """Get a provider instance by name.

    Args:
        provider_name: "ollama", "openai", or "gemini"
        **kwargs: Provider-specific arguments (model_name, n_samples, etc.)

    Returns:
        Provider instance
    """
    if provider_name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(PROVIDERS.keys())}")

    return PROVIDERS[provider_name](**kwargs)


def list_providers() -> List[str]:
    """List available provider names."""
    return list(PROVIDERS.keys())


# Quick test
if __name__ == "__main__":
    import sys

    provider_name = sys.argv[1] if len(sys.argv) > 1 else "ollama"

    print(f"Testing provider: {provider_name}")
    print("=" * 50)

    try:
        if provider_name == "ollama":
            provider = get_provider("ollama", model_name="llama3.1:latest", n_samples=10)
        elif provider_name == "openai":
            provider = get_provider("openai", model_name="gpt-4o-mini")
        elif provider_name == "gemini":
            provider = get_provider("gemini", model_name="gemini-2.0-flash")
        else:
            provider = get_provider(provider_name)

        prompt = """You are answering a multiple choice question.

Question:
What is 2 + 2?

Options:
A) 3
B) 4
C) 5
D) 6

Answer with A, B, C, or D:"""

        print(f"Model: {provider.model_name}")
        print(f"Supports logprobs: {provider.supports_logprobs}")
        print()

        result = provider.get_action_probs(prompt)
        print(f"Method: {result.method}")
        print(f"Probs: {' '.join(f'{k}:{v:.3f}' for k, v in sorted(result.probs.items()))}")
        print(f"Entropy: {result.entropy:.3f} bits")
        if result.raw_logprobs:
            print(f"Raw logprobs: {result.raw_logprobs}")

    except Exception as e:
        print(f"Error: {e}")
