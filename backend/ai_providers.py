"""
Multi-AI Provider System for Browser Automation
Supports Groq, OpenAI, Anthropic, and Google AI with fallback chains
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import base64

# Import AI clients
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from google.generativeai import GenerativeModel
    import google.generativeai as genai
except ImportError:
    genai = None
    GenerativeModel = None


class AIProvider(Enum):
    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class AIResponse:
    content: str
    provider: str
    model: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = None


class BaseAIProvider(ABC):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        
    @abstractmethod
    async def generate_code(self, command: str, dom: str = "", screenshot: str = "") -> AIResponse:
        pass
    
    @abstractmethod
    async def analyze_screenshot(self, screenshot: str, command: str) -> AIResponse:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class GroqProvider(BaseAIProvider):
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        super().__init__(api_key, model)
        if Groq:
            self.client = Groq(api_key=api_key)
        else:
            self.client = None
    
    def is_available(self) -> bool:
        return Groq is not None and self.client is not None
    
    async def generate_code(self, command: str, dom: str = "", screenshot: str = "") -> AIResponse:
        if not self.is_available():
            raise RuntimeError("Groq client not available")
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert browser automation assistant that converts natural language "
                    "commands into Playwright Python code. Return only executable code without "
                    "markdown fences or explanations. Use await for all async operations. "
                    "The 'page' object is available in context."
                )
            },
            {
                "role": "user", 
                "content": f"Command: {command}\n\nDOM Context:\n{dom}" if dom else command
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=1000
            )
            
            code = response.choices[0].message.content.strip()
            
            # Clean markdown fences
            if code.startswith("```"):
                code = code.split("```")[1]
                if code.startswith("python"):
                    code = code[len("python"):].strip()
                code = code.split("```")[0].strip()
            
            return AIResponse(
                content=code,
                provider="groq",
                model=self.model,
                confidence=0.8
            )
        except Exception as e:
            raise RuntimeError(f"Groq API error: {str(e)}")
    
    async def analyze_screenshot(self, screenshot: str, command: str) -> AIResponse:
        # Groq doesn't support vision yet, return basic response
        return AIResponse(
            content="Vision analysis not supported by Groq",
            provider="groq",
            model=self.model,
            confidence=0.0
        )


class OpenAIProvider(BaseAIProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        super().__init__(api_key, model)
        if openai:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = None
    
    def is_available(self) -> bool:
        return openai is not None and self.client is not None
    
    async def generate_code(self, command: str, dom: str = "", screenshot: str = "") -> AIResponse:
        if not self.is_available():
            raise RuntimeError("OpenAI client not available")
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert browser automation assistant. Convert natural language "
                    "commands into executable Playwright Python code. Return only code without "
                    "markdown formatting. Use await for async operations. 'page' object is available."
                )
            }
        ]
        
        # Add content based on available context
        user_content = f"Command: {command}"
        if dom:
            user_content += f"\n\nDOM Context:\n{dom}"
        
        if screenshot:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_content},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"}
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": user_content})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=1000
            )
            
            code = response.choices[0].message.content.strip()
            
            # Clean markdown fences
            if code.startswith("```"):
                code = code.split("```")[1]
                if code.startswith("python"):
                    code = code[len("python"):].strip()
                code = code.split("```")[0].strip()
            
            return AIResponse(
                content=code,
                provider="openai",
                model=self.model,
                confidence=0.9
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
    
    async def analyze_screenshot(self, screenshot: str, command: str) -> AIResponse:
        if not self.is_available():
            raise RuntimeError("OpenAI client not available")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Analyze this screenshot and provide coordinates for the element mentioned in the command. Return JSON with x, y coordinates and element description."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Find element for command: {command}"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"}
                            }
                        ]
                    }
                ],
                temperature=0,
                max_tokens=500
            )
            
            return AIResponse(
                content=response.choices[0].message.content,
                provider="openai",
                model=self.model,
                confidence=0.85
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI vision analysis error: {str(e)}")


class AnthropicProvider(BaseAIProvider):
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        super().__init__(api_key, model)
        if anthropic:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = None
    
    def is_available(self) -> bool:
        return anthropic is not None and self.client is not None
    
    async def generate_code(self, command: str, dom: str = "", screenshot: str = "") -> AIResponse:
        if not self.is_available():
            raise RuntimeError("Anthropic client not available")
        
        content = f"Command: {command}"
        if dom:
            content += f"\n\nDOM Context:\n{dom}"
        
        messages = [
            {
                "role": "user",
                "content": content if not screenshot else [
                    {"type": "text", "text": content},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": screenshot
                        }
                    }
                ]
            }
        ]
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0,
                system=(
                    "You are an expert browser automation assistant. Convert natural language "
                    "commands into executable Playwright Python code. Return only code without "
                    "markdown formatting. Use await for async operations. 'page' object is available."
                ),
                messages=messages
            )
            
            code = response.content[0].text.strip()
            
            # Clean markdown fences
            if code.startswith("```"):
                code = code.split("```")[1]
                if code.startswith("python"):
                    code = code[len("python"):].strip()
                code = code.split("```")[0].strip()
            
            return AIResponse(
                content=code,
                provider="anthropic",
                model=self.model,
                confidence=0.9
            )
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")
    
    async def analyze_screenshot(self, screenshot: str, command: str) -> AIResponse:
        if not self.is_available():
            raise RuntimeError("Anthropic client not available")
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0,
                system="Analyze this screenshot and provide coordinates for the element mentioned in the command. Return JSON with x, y coordinates and element description.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Find element for command: {command}"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": screenshot
                                }
                            }
                        ]
                    }
                ]
            )
            
            return AIResponse(
                content=response.content[0].text,
                provider="anthropic",
                model=self.model,
                confidence=0.85
            )
        except Exception as e:
            raise RuntimeError(f"Anthropic vision analysis error: {str(e)}")


class AIProviderManager:
    def __init__(self):
        self.providers: Dict[str, BaseAIProvider] = {}
        self.fallback_chain: List[str] = []
        self.setup_providers()
    
    def setup_providers(self):
        """Initialize available AI providers"""
        # Groq
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                self.providers["groq"] = GroqProvider(groq_key)
                if self.providers["groq"].is_available():
                    self.fallback_chain.append("groq")
            except Exception:
                pass
        
        # OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                self.providers["openai"] = OpenAIProvider(openai_key)
                if self.providers["openai"].is_available():
                    self.fallback_chain.append("openai")
            except Exception:
                pass
        
        # Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                self.providers["anthropic"] = AnthropicProvider(anthropic_key)
                if self.providers["anthropic"].is_available():
                    self.fallback_chain.append("anthropic")
            except Exception:
                pass
        
        print(f"Available AI providers: {self.fallback_chain}")
    
    async def generate_code_with_fallback(self, command: str, dom: str = "", screenshot: str = "") -> AIResponse:
        """Try multiple providers with fallback chain"""
        last_error = None
        
        for provider_name in self.fallback_chain:
            provider = self.providers.get(provider_name)
            if not provider or not provider.is_available():
                continue
            
            try:
                print(f"Trying {provider_name} for code generation...")
                result = await provider.generate_code(command, dom, screenshot)
                print(f"✅ {provider_name} succeeded")
                return result
            except Exception as e:
                print(f"❌ {provider_name} failed: {str(e)}")
                last_error = e
                continue
        
        # All providers failed
        raise RuntimeError(f"All AI providers failed. Last error: {str(last_error)}")
    
    async def analyze_screenshot_with_fallback(self, screenshot: str, command: str) -> AIResponse:
        """Try vision-capable providers for screenshot analysis"""
        vision_providers = ["openai", "anthropic"]  # Providers that support vision
        last_error = None
        
        for provider_name in vision_providers:
            if provider_name not in self.providers:
                continue
                
            provider = self.providers[provider_name]
            if not provider.is_available():
                continue
            
            try:
                print(f"Trying {provider_name} for vision analysis...")
                result = await provider.analyze_screenshot(screenshot, command)
                print(f"✅ {provider_name} vision analysis succeeded")
                return result
            except Exception as e:
                print(f"❌ {provider_name} vision analysis failed: {str(e)}")
                last_error = e
                continue
        
        raise RuntimeError(f"All vision providers failed. Last error: {str(last_error)}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return self.fallback_chain.copy()


# Global instance
ai_manager = AIProviderManager()