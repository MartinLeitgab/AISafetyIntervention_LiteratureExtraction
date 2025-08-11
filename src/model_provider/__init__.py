"""Model provider package for SOAR-5 project.

This package provides a unified interface for AI model providers,
with comprehensive OpenAI integration including batch processing support.
"""

from .openai_provider import OpenAIModelProvider

__all__ = [
    "OpenAIModelProvider"
]

__version__ = "2.0.0"
