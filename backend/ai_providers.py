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

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


class AIProvider(Enum):
    STARCODER2 = "starcoder2"
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
                    "markdown formatting. Use await for async operations. 'page' object is available.\n\n"
                    "IMPORTANT E-COMMERCE GUIDELINES:\n"
                    "- For 'Add' button clicks, use multiple selector strategies:\n"
                    "  1. Try class-based: button[class*='add'], button[class*='cta'], button.btn.hand.cta-add\n"
                    "  2. Try data attributes: *[data-testid*='add'], *[data-action*='add']\n"
                    "  3. Try text patterns: button:has-text('Add'), button:has-text('+')\n"
                    "- Always scroll elements into view before clicking\n"
                    "- Use hover before clicking for interactive buttons\n"
                    "- Include retry logic with multiple selectors\n\n"
                    "Example for 'Add pizza' command:\n"
                    "```python\n"
                    "# Try multiple strategies for Add button\n"
                    "selectors = [\n"
                    "    'button[class*=\"cta-add\"]',\n"
                    "    'button.btn.hand.cta-add',\n"
                    "    'button:has-text(\"Add\")',\n"
                    "    '*[data-testid*=\"add\"]'\n"
                    "]\n\n"
                    "for selector in selectors:\n"
                    "    try:\n"
                    "        element = await page.wait_for_selector(selector, timeout=3000)\n"
                    "        if element and await element.is_visible():\n"
                    "            await element.scroll_into_view_if_needed()\n"
                    "            await element.hover()\n"
                    "            await element.click()\n"
                    "            break\n"
                    "    except:\n"
                    "        continue\n"
                    "```"
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


class StarCoder2Provider(BaseAIProvider):
    def __init__(self, model_name: str = "bigcode/starcoder2-3b"):
        super().__init__("", model_name)  # No API key needed for local model
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        self._loading = False
        
    def is_available(self) -> bool:
        """Check if required packages are available"""
        return (torch is not None and 
                AutoModelForCausalLM is not None and 
                AutoTokenizer is not None)
    
    async def _load_model(self):
        """Load the model and tokenizer if not already loaded"""
        if self.model is not None or self._loading:
            return
            
        self._loading = True
        try:
            print(f"🔥 Loading StarCoder2 model: {self.model_name}")
            
            # Detect device
            if torch.cuda.is_available():
                self.device = "cuda"
                print("🚀 Using GPU acceleration")
            else:
                self.device = "cpu"
                print("💻 Using CPU (consider GPU for faster inference)")
            
            # Load tokenizer
            print("📚 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            print("🧠 Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print("✅ StarCoder2 model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load StarCoder2 model: {str(e)}")
            self.model = None
            self.tokenizer = None
            raise
        finally:
            self._loading = False
    
    async def generate_code(self, command: str, dom: str = "", screenshot: str = "") -> AIResponse:
        if not self.is_available():
            raise RuntimeError("StarCoder2 dependencies not available")
        
        await self._load_model()
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("StarCoder2 model not loaded")
        
        # Create specialized prompt for StarCoder2
        prompt = self._create_starcoder_prompt(command, dom)
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncate=True, max_length=2048)
            inputs = inputs.to(self.device)
            
            # Generate code
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    stop_strings=["```", "\n\n#", "def ", "class "]
                )
            
            # Decode response
            generated = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            code = self._clean_starcoder_output(generated)
            
            return AIResponse(
                content=code,
                provider="starcoder2",
                model=self.model_name,
                confidence=0.9
            )
            
        except Exception as e:
            raise RuntimeError(f"StarCoder2 generation error: {str(e)}")
    
    def _create_starcoder_prompt(self, command: str, dom: str = "") -> str:
        """Create optimized prompt for StarCoder2"""
        prompt = f"""# Playwright Python automation code
# Task: {command}
# Generate clean, executable Playwright code

import asyncio
from playwright.async_api import async_playwright

async def automate(page):
    # {command}
"""
        
        if dom:
            # Truncate DOM if too long
            dom_snippet = dom[:1000] + "..." if len(dom) > 1000 else dom
            prompt += f"""
    # DOM context (partial):
    # {dom_snippet}
"""
        
        prompt += """
    # Multiple selector strategies for robust clicking
    selectors = [
        'button[class*="cta-add"]',
        'button[class*="add"]', 
        'button.btn.hand.cta-add',
        'button:has-text("Add")',
        '*[data-testid*="add"]'
    ]
    
    for selector in selectors:
        try:
            element = await page.wait_for_selector(selector, timeout=3000)
            if element and await element.is_visible():
                await element.scroll_into_view_if_needed()
                await element.hover()
                await element.click()
                break
        except:
            continue
"""
        return prompt
    
    def _clean_starcoder_output(self, generated: str) -> str:
        """Clean and extract code from StarCoder2 output"""
        # Remove any markdown fences
        if "```" in generated:
            parts = generated.split("```")
            for part in parts:
                if "await" in part or "page." in part:
                    generated = part.strip()
                    break
        
        # Extract just the automation code (remove imports, function defs)
        lines = generated.split('\n')
        code_lines = []
        in_function = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('async def') or stripped.startswith('def'):
                in_function = True
                continue
            elif in_function and (stripped.startswith('await') or stripped.startswith('try:') or 
                                stripped.startswith('for ') or stripped.startswith('selectors')):
                # Extract indented content
                code_lines.append(line[4:] if line.startswith('    ') else line)
            elif not in_function and (stripped.startswith('await') or 'page.' in stripped):
                code_lines.append(line)
        
        return '\n'.join(code_lines).strip()
    
    async def analyze_screenshot(self, screenshot: str, command: str) -> AIResponse:
        # StarCoder2 doesn't support vision, delegate to next provider
        return AIResponse(
            content="Vision analysis not supported by StarCoder2",
            provider="starcoder2", 
            model=self.model_name,
            confidence=0.0
        )


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
        
        # StarCoder2 (Priority 1 - Local, free, unlimited, specialized for code)
        print(f"⭐ Attempting StarCoder2 setup...")
        try:
            starcoder2_provider = StarCoder2Provider()
            if starcoder2_provider.is_available():
                self.providers["starcoder2"] = starcoder2_provider
                self.fallback_chain.append("starcoder2")
                print(f"✅ StarCoder2 provider added successfully (local model)")
            else:
                print(f"❌ StarCoder2 dependencies not available (torch/transformers missing)")
        except Exception as e:
            print(f"❌ StarCoder2 setup failed: {str(e)}")
        
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
        
        # Ollama (Priority 3 - Free, unlimited, local, but slower)
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
        
        # OpenAI (Priority 4 - Reliable, expensive)
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
        
        # Anthropic (Priority 5 - Reliable, expensive)
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
    
    async def generate_code_with_fallback(self, command: str, dom: str = "", screenshot: str = "", skip_providers: List[str] = None) -> AIResponse:
        """Try multiple providers with fallback chain, optionally skipping failed providers"""
        last_error = None
        skip_providers = skip_providers or []
        
        for provider_name in self.fallback_chain:
            provider = self.providers.get(provider_name)
            if not provider or not provider.is_available() or provider_name in skip_providers:
                if provider_name in skip_providers:
                    print(f"⏭️  Skipping {provider_name} (already failed)")
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
    
    async def analyze_screenshot_with_fallback(self, screenshot: str, command: str, skip_providers: List[str] = None) -> AIResponse:
        """Try vision-capable providers for screenshot analysis, optionally skipping failed providers"""
        vision_providers = ["openai", "anthropic"]  # Providers that support vision
        last_error = None
        skip_providers = skip_providers or []
        
        for provider_name in vision_providers:
            if provider_name not in self.providers or provider_name in skip_providers:
                if provider_name in skip_providers:
                    print(f"⏭️  Skipping {provider_name} vision (already failed)")
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