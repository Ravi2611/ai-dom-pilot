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

try:
    import ollama
except ImportError:
    ollama = None


class AIProvider(Enum):
    OLLAMA = "ollama"
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
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
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
    def __init__(self, api_key: str, model: str = "claude-3-5-haiku-20241022"):
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


class OllamaProvider(BaseAIProvider):
    def __init__(self, host: str = "localhost:11434", model: str = "codellama:7b"):
        super().__init__("", model)  # Ollama doesn't use API keys
        self.host = host
        self.client = ollama if ollama else None
    
    def is_available(self) -> bool:
        if not self.client:
            print(f"❌ Ollama client not available - ollama package not installed?")
            return False
        try:
            # Test connection and model availability
            print(f"🔍 Testing Ollama connection to {self.host}...")
            response = self.client.list()
            print(f"🐛 Raw Ollama response: {response}")
            
            # Handle different possible response formats
            models = []
            if isinstance(response, dict):
                if 'models' in response:
                    # Handle dict response with 'models' key
                    for model in response['models']:
                        if isinstance(model, dict) and 'name' in model:
                            models.append(model['name'])
                        elif isinstance(model, dict) and 'model' in model:
                            models.append(model['model'])
                        elif isinstance(model, str):
                            models.append(model)
                elif hasattr(response, 'models'):
                    # Handle object response with models attribute
                    for model in response.models:
                        if hasattr(model, 'name'):
                            models.append(model.name)
                        elif hasattr(model, 'model'):
                            models.append(model.model)
                        elif isinstance(model, str):
                            models.append(model)
            elif hasattr(response, 'models'):
                # Handle response object with models attribute
                for model in response.models:
                    if hasattr(model, 'name'):
                        models.append(model.name)
                    elif hasattr(model, 'model'):
                        models.append(model.model)
                    elif isinstance(model, str):
                        models.append(model)
            
            print(f"📋 Extracted model names: {models}")
            if self.model in models:
                print(f"✅ Model {self.model} found!")
                return True
            else:
                print(f"❌ Model {self.model} not found in available models")
                print(f"💡 Available models: {models}")
                return False
        except Exception as e:
            print(f"❌ Ollama connection error: {str(e)}")
            print(f"🐛 Error type: {type(e).__name__}")
            import traceback
            print(f"🐛 Full traceback: {traceback.format_exc()}")
            return False
    
    async def generate_code(self, command: str, dom: str = "", screenshot: str = "") -> AIResponse:
        if not self.is_available():
            raise RuntimeError("Ollama client not available or model not found")
        
        prompt = f"""TASK: Generate Playwright Python code for: {command}

RULES:
1. Output ONLY executable Python code
2. NO explanations, markdown, or comments
3. Use 'page' variable (Playwright Page object)
4. Use await for async operations
5. Start coding immediately

EXAMPLES:
- Click button: await page.click('button:has-text("Submit")')
- Fill input: await page.fill('input[name="email"]', "test@example.com")
- Wait: await page.wait_for_timeout(1000)

CODE:"""
        
        if dom:
            prompt += f"\n\nDOM Context:\n{dom}"
        
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': 0,
                    'num_predict': 1000,
                    'stop': ['```', 'explanation:', 'note:']
                }
            )
            
            code = response['response'].strip()
            
            # Clean any markdown fences that might slip through
            if code.startswith("```"):
                code = code.split("```")[1]
                if code.startswith("python"):
                    code = code[len("python"):].strip()
                code = code.split("```")[0].strip()
            
            return AIResponse(
                content=code,
                provider="ollama",
                model=self.model,
                confidence=0.85
            )
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {str(e)}")
    
    async def analyze_screenshot(self, screenshot: str, command: str) -> AIResponse:
        # Ollama doesn't support vision analysis for most models
        return AIResponse(
            content="Vision analysis not supported by Ollama",
            provider="ollama",
            model=self.model,
            confidence=0.0
        )


class AIProviderManager:
    def __init__(self):
        self.providers: Dict[str, BaseAIProvider] = {}
        self.fallback_chain: List[str] = []
        self.setup_providers()
    
    def setup_providers(self):
        """Initialize available AI providers"""
        print("🔧 Setting up AI providers...")
        
        # Ollama (Priority 1 - Free, unlimited, local)
        ollama_host = os.getenv("OLLAMA_HOST", "localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "codellama:7b")
        print(f"🐪 Attempting Ollama setup: {ollama_host} with model {ollama_model}")
        try:
            self.providers["ollama"] = OllamaProvider(ollama_host, ollama_model)
            if self.providers["ollama"].is_available():
                self.fallback_chain.append("ollama")
                print(f"✅ Ollama provider added successfully with model {ollama_model}")
            else:
                print(f"❌ Ollama not available - server running? Model {ollama_model} exists?")
                print(f"💡 Check: docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama")
        except Exception as e:
            print(f"❌ Ollama setup failed: {str(e)}")
        
        # Groq (Priority 2 - Fast, rate limited)
        groq_key = os.getenv("GROQ_API_KEY")
        print(f"🚀 Attempting Groq setup: API key {'✅ provided' if groq_key else '❌ missing'}")
        if groq_key:
            try:
                self.providers["groq"] = GroqProvider(groq_key)
                if self.providers["groq"].is_available():
                    self.fallback_chain.append("groq")
                    print(f"✅ Groq provider added successfully")
                else:
                    print(f"❌ Groq provider not available (API key invalid?)")
            except Exception as e:
                print(f"❌ Groq setup failed: {str(e)}")
        else:
            print("⚠️ Groq API key not found in environment")
        
        # OpenAI (Priority 3 - Reliable, expensive)
        openai_key = os.getenv("OPENAI_API_KEY")
        print(f"🤖 Attempting OpenAI setup: API key {'✅ provided' if openai_key else '❌ missing'}")
        if openai_key:
            try:
                self.providers["openai"] = OpenAIProvider(openai_key)
                if self.providers["openai"].is_available():
                    self.fallback_chain.append("openai")
                    print(f"✅ OpenAI provider added successfully")
                else:
                    print(f"❌ OpenAI provider not available")
            except Exception as e:
                print(f"❌ OpenAI setup failed: {str(e)}")
        
        # Anthropic (Priority 4 - Reliable, expensive)
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        print(f"🧠 Attempting Anthropic setup: API key {'✅ provided' if anthropic_key else '❌ missing'}")
        if anthropic_key:
            try:
                self.providers["anthropic"] = AnthropicProvider(anthropic_key)
                if self.providers["anthropic"].is_available():
                    self.fallback_chain.append("anthropic")
                    print(f"✅ Anthropic provider added successfully")
                else:
                    print(f"❌ Anthropic provider not available")
            except Exception as e:
                print(f"❌ Anthropic setup failed: {str(e)}")
        
        print(f"🎯 Final fallback chain: {self.fallback_chain}")
        if not self.fallback_chain:
            print("⚠️ WARNING: No AI providers available! Check your configuration.")
        elif len(self.fallback_chain) == 1:
            print(f"⚠️ WARNING: Only one provider ({self.fallback_chain[0]}) available. Consider adding fallbacks.")
    
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