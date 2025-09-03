from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import os
import base64
import sqlite3
import json
from datetime import datetime
from typing import Optional, List
import traceback
from playwright.async_api import async_playwright
from groq import Groq
import re
import textwrap
from bs4 import BeautifulSoup, Comment
from websocket_browser import websocket_endpoint, browser_manager, get_shared_browser
from contextlib import asynccontextmanager

# Import new AI and automation systems
from ai_providers import ai_manager
from smart_selectors import SmartRetrySystem
from vision_automation import VisionAutomationEngine, VisionCommandGenerator

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    await browser_manager.cleanup()

app = FastAPI(title="AI Browser Automation API", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client (you'll need to set GROQ_API_KEY env variable)
client = Groq(api_key=os.getenv("GROQ_API_KEY", "your-groq-api-key-here"))

# Database setup
def init_db():
    conn = sqlite3.connect('automation.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS commands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            command TEXT NOT NULL,
            generated_code TEXT,
            status TEXT DEFAULT 'pending',
            screenshot_path TEXT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Create screenshots directory
os.makedirs("screenshots", exist_ok=True)

# Pydantic models
class AutomationCommand(BaseModel):
    command: str
    dom: Optional[str] = ""

class CommandResponse(BaseModel):
    id: int
    command: str
    generated_code: str
    status: str
    screenshot_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str

# Global browser instance (removed - now using browser_manager)

def shrink_dom(dom: str) -> str:
    """Compress DOM by keeping only actionable elements - Enhanced version"""
    if not dom:
        return ""
    
    soup = BeautifulSoup(dom, "html.parser")
    
    # Remove scripts, styles, comments
    for tag in soup(["script", "style"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    
    # Enhanced: Keep more interactive elements including divs with roles
    actionable_elements = soup.find_all([
        "form", "input", "button", "a", "select", "textarea", "label",
        # Enhanced: Include divs and spans that might be interactive
        "div[role]", "span[role]", "div[onclick]", "span[onclick]",
        "div[data-testid]", "div[class*='button']", "div[class*='click']",
        "div[class*='radio']", "div[class*='checkbox']", "span[class*='button']"
    ])
    
    # Also include elements with specific attributes that suggest interactivity
    interactive_attrs = ['onclick', 'onchange', 'role', 'tabindex', 'data-testid', 'aria-label']
    for attr in interactive_attrs:
        elements_with_attr = soup.find_all(attrs={attr: True})
        actionable_elements.extend(elements_with_attr)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_elements = []
    for el in actionable_elements:
        el_str = str(el)
        if el_str not in seen:
            seen.add(el_str)
            unique_elements.append(el)
    
    return "\n".join(str(el) for el in unique_elements[:50])  # Limit to 50 elements

async def ai_to_code(command: str, dom: str = "") -> str:
    """Enhanced AI code generation with multi-provider fallback and smart retry"""
    command_lower = command.lower()
    
    # Extract OTP digits from command
    otp_match = re.search(r"\b\d{4,6}\b", command)
    otp_value = otp_match.group(0) if otp_match else "987654"
    
    # Multi-box OTP
    if "otp" in command_lower and ("box" in command_lower or "placeholder" in command_lower or "field" in command_lower):
        return f'''
otp = "{otp_value}"
inputs = await page.query_selector_all('input[type="text"], input[type="number"]')
otp_inputs = [inp for inp in inputs if await inp.get_attribute('maxlength') == '1' or 'otp' in (await inp.get_attribute('name') or '').lower()]
for i, digit in enumerate(otp):
    if i < len(otp_inputs):
        await otp_inputs[i].fill(digit)
        await page.wait_for_timeout(100)
'''
    
    # Single input OTP
    if "otp" in command_lower:
        return f'await page.fill(\'input[name="otp"], input[placeholder*="otp" i]\', "{otp_value}")'
    
    # Use multi-provider AI system with vision enhancement
    try:
        # Check if we should use vision-enhanced generation
        if browser_manager.page:
            vision_generator = VisionCommandGenerator(browser_manager.page)
            code = await vision_generator.generate_vision_enhanced_code(command, dom)
            
            # Add smart retry wrapper with AI fallback
            enhanced_code = f'''
# Enhanced automation with smart retry system and AI fallback
from smart_selectors import SmartRetrySystem
from vision_automation import VisionAutomationEngine

retry_system = SmartRetrySystem(page)
vision_engine = VisionAutomationEngine(page)

# Track tried providers to avoid infinite loops
_tried_providers = getattr(page, '_tried_providers', set())
_retry_count = getattr(page, '_retry_count', 0)

# Original generated code with smart fallbacks
try:
{textwrap.indent(code, '    ')}
except Exception as e:
    print(f"Primary automation failed: {{e}}")
    
    # Smart retry for click operations
    if "click" in "{command_lower}":
        target_text = "{command}".split("click")[-1].strip().strip("on").strip()
        if target_text:
            success = await retry_system.smart_click(target_text, """{dom}""")
            if not success:
                success = await vision_engine.vision_click("{command}")
                if not success:
                    # AI model fallback - regenerate code with next provider
                    await _ai_model_fallback(page, "{command}", """{dom}""", _tried_providers, _retry_count)
    
    # Smart retry for fill operations  
    elif "fill" in "{command_lower}" or "enter" in "{command_lower}" or "type" in "{command_lower}":
        parts = "{command}".split()
        if len(parts) >= 2:
            field_name = parts[0]
            value = " ".join(parts[1:])
            success = await retry_system.smart_fill(field_name, value, """{dom}""")
            if not success:
                success = await vision_engine.vision_fill("{command}", value)
                if not success:
                    # AI model fallback - regenerate code with next provider
                    await _ai_model_fallback(page, "{command}", """{dom}""", _tried_providers, _retry_count)
    else:
        # Direct AI model fallback for other operations
        await _ai_model_fallback(page, "{command}", """{dom}""", _tried_providers, _retry_count)
'''
            return enhanced_code
        else:
            # Fallback to regular AI generation
            response = await ai_manager.generate_code_with_fallback(command, dom)
            return response.content
            
    except Exception as e:
        print(f"Enhanced AI generation failed: {e}")
        # Use the AI manager's fallback system instead of redundant Groq fallback
        try:
            print("🔄 Falling back to AI manager's fallback system...")
            response = await ai_manager.generate_code_with_fallback(command, dom)
            return response.content
        except Exception as fallback_error:
            print(f"❌ All AI providers failed: {str(fallback_error)}")
            return f'# Error: All AI providers failed - {str(fallback_error)}\n# Please check your AI provider configuration\nawait page.wait_for_timeout(1000)'

async def _ai_model_fallback(page, command: str, dom: str, tried_providers: set, retry_count: int):
    """AI model fallback function for failed automation actions"""
    
    # Prevent infinite loops
    max_retries = 3
    if retry_count >= max_retries:
        print(f"❌ Maximum retry attempts ({max_retries}) reached for command: {command}")
        raise Exception(f"All retry attempts exhausted for command: {command}")
    
    # Update page state
    page._retry_count = retry_count + 1
    page._tried_providers = tried_providers
    
    print(f"🔄 AI Model Fallback (attempt {retry_count + 1}/{max_retries}) for command: {command}")
    
    try:
        # Get current page DOM for context
        current_dom = dom
        if not current_dom and page:
            try:
                current_dom = await page.content()
                current_dom = shrink_dom(current_dom)
            except:
                pass
        
        # Generate new code with next AI provider
        response = await ai_manager.generate_code_with_fallback(command, current_dom)
        
        if response and response.content:
            print(f"✅ Generated fallback code with {response.provider}: {response.content[:100]}...")
            
            # Execute the newly generated code
            try:
                # Create isolated execution context
                exec_code = f"""
async def fallback_execute():
{textwrap.indent(response.content, '    ')}
"""
                
                fallback_globals = {
                    "__builtins__": __builtins__,
                    "page": page
                }
                
                exec(compile(exec_code, "<string>", "exec"), fallback_globals)
                fallback_function = fallback_globals['fallback_execute']
                await fallback_function()
                
                print(f"✅ Fallback code executed successfully with {response.provider}")
                return True
                
            except Exception as exec_error:
                print(f"⚠️ Fallback code execution failed: {exec_error}")
                # Try the next provider recursively
                return await _ai_model_fallback(page, command, dom, tried_providers, retry_count + 1)
        else:
            print("❌ No fallback code generated")
            raise Exception("Failed to generate fallback code")
            
    except Exception as e:
        print(f"❌ AI model fallback failed: {e}")
        if retry_count < max_retries - 1:
            return await _ai_model_fallback(page, command, dom, tried_providers, retry_count + 1)
        else:
            raise Exception(f"All AI model fallback attempts failed: {e}")

async def take_screenshot(command_id: int) -> str:
    """Take screenshot and save it"""
    try:
        os.makedirs("screenshots", exist_ok=True)
        screenshot_path = f"screenshots/step_{command_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        # Use browser_manager.page for screenshot
        if browser_manager.page:
            await browser_manager.page.screenshot(path=screenshot_path)
            return screenshot_path
        return ""
    except Exception as e:
        print(f"Screenshot error: {e}")
        return ""

async def execute_automation_command(command_id: int, command: str, generated_code: str):
    """Execute the automation command in background"""
    conn = sqlite3.connect('automation.db')
    
    try:
        # Initialize browser through browser_manager if needed
        if not (browser_manager.browser and browser_manager.browser.is_connected() and browser_manager.page):
            await browser_manager.initialize_browser()
        
        # Update status to running
        conn.execute("UPDATE commands SET status = 'running' WHERE id = ?", (command_id,))
        conn.commit()
        
        # Execute the generated code
        print(f"Executing code for command {command_id}:")
        print(generated_code)
        
        # Create async execution context using browser_manager.page
        async_code = f"""
async def execute_command():
{textwrap.indent(generated_code, '    ')}
"""
        
        # Execute the async code to define the function with AI fallback context
        exec_globals = {
            "__builtins__": __builtins__, 
            "page": browser_manager.page,
            "_ai_model_fallback": _ai_model_fallback,
            "ai_manager": ai_manager,
            "textwrap": textwrap
        }
        exec(compile(async_code, "<string>", "exec"), exec_globals)
        
        # Get the function and execute it in our async context
        execute_command = exec_globals['execute_command']
        await execute_command()
        
        # Take screenshot
        screenshot_path = await take_screenshot(command_id)
        
        # Force immediate screenshot update in WebSocket stream
        if browser_manager.active_connections:
            screenshot_b64 = await browser_manager.take_screenshot()
            if screenshot_b64:
                await browser_manager.send_frame_update_with_screenshot(screenshot_b64)
                print(f"✅ Screenshot updated after command {command_id}")
            else:
                print(f"⚠️ Failed to take screenshot after command {command_id}")
        else:
            print(f"ℹ️ No active WebSocket connections for command {command_id}")
        
        # Update status to success
        conn.execute(
            "UPDATE commands SET status = 'success', screenshot_path = ? WHERE id = ?",
            (screenshot_path, command_id)
        )
        conn.commit()
        
    except Exception as e:
        error_msg = str(e)
        print(f"Execution error: {error_msg}")
        print(traceback.format_exc())
        
        # Update status to error
        conn.execute(
            "UPDATE commands SET status = 'error', error_message = ? WHERE id = ?",
            (error_msg, command_id)
        )
        conn.commit()
    
    finally:
        conn.close()

@app.post("/api/automation/command", response_model=CommandResponse)
async def create_automation_command(command: AutomationCommand, background_tasks: BackgroundTasks):
    """Create and execute an automation command"""
    conn = sqlite3.connect('automation.db')
    
    try:
        # Get current DOM if page exists
        current_dom = ""
        if browser_manager.page:
            try:
                current_dom = await browser_manager.page.content()
                current_dom = shrink_dom(current_dom)
            except:
                pass
        
        # Use provided DOM or current DOM
        dom_to_use = command.dom if command.dom else current_dom
        
        # Generate code with enhanced AI system
        generated_code = await ai_to_code(command.command, dom_to_use)
        
        # Insert command into database
        cursor = conn.execute(
            "INSERT INTO commands (command, generated_code, status) VALUES (?, ?, 'pending')",
            (command.command, generated_code)
        )
        command_id = cursor.lastrowid
        conn.commit()
        
        # Execute in background
        background_tasks.add_task(execute_automation_command, command_id, command.command, generated_code)
        
        return CommandResponse(
            id=command_id,
            command=command.command,
            generated_code=generated_code,
            status="pending",
            created_at=datetime.now().isoformat()
        )
    
    finally:
        conn.close()

@app.get("/api/automation/commands", response_model=List[CommandResponse])
async def get_commands():
    """Get all automation commands"""
    conn = sqlite3.connect('automation.db')
    
    try:
        cursor = conn.execute(
            "SELECT id, command, generated_code, status, screenshot_path, error_message, created_at FROM commands ORDER BY created_at DESC LIMIT 50"
        )
        commands = []
        
        for row in cursor.fetchall():
            screenshot_url = f"/screenshots/{os.path.basename(row[4])}" if row[4] else None
            
            commands.append(CommandResponse(
                id=row[0],
                command=row[1],
                generated_code=row[2],
                status=row[3],
                screenshot_url=screenshot_url,
                error_message=row[5],
                created_at=row[6]
            ))
        
        return commands
    
    finally:
        conn.close()

@app.get("/api/automation/command/{command_id}", response_model=CommandResponse)
async def get_command(command_id: int):
    """Get a specific command by ID"""
    conn = sqlite3.connect('automation.db')
    
    try:
        cursor = conn.execute(
            "SELECT id, command, generated_code, status, screenshot_path, error_message, created_at FROM commands WHERE id = ?",
            (command_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Command not found")
        
        screenshot_url = f"/screenshots/{os.path.basename(row[4])}" if row[4] else None
        
        return CommandResponse(
            id=row[0],
            command=row[1],
            generated_code=row[2],
            status=row[3],
            screenshot_url=screenshot_url,
            error_message=row[5],
            created_at=row[6]
        )
    
    finally:
        conn.close()

@app.get("/api/automation/browser/current-url")
async def get_current_url():
    """Get current browser URL"""
    if browser_manager.page:
        try:
            return {"url": browser_manager.page.url}
        except:
            return {"url": "about:blank"}
    return {"url": "about:blank"}

@app.get("/api/automation/browser/dom")
async def get_current_dom():
    """Get current page DOM (compressed)"""
    if browser_manager.page:
        try:
            dom = await browser_manager.page.content()
            return {"dom": shrink_dom(dom)}
        except Exception as e:
            return {"dom": "", "error": str(e)}
    return {"dom": ""}

# WebSocket endpoint for browser streaming
@app.websocket("/ws/browser")
async def websocket_browser_endpoint(websocket: WebSocket):
    await websocket_endpoint(websocket)

@app.post("/api/automation/reset")
async def reset_browser():
    """Reset browser session and clear automation state"""
    try:
        await browser_manager.reset_browser()
        # Clear database records if needed (optional)
        conn = sqlite3.connect('automation.db')
        try:
            conn.execute("DELETE FROM commands")
            conn.commit()
        finally:
            conn.close()
        
        return {"status": "success", "message": "Browser and automation state reset successfully"}
    except Exception as e:
        print(f"Error resetting automation: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

# Serve screenshots
app.mount("/screenshots", StaticFiles(directory="screenshots"), name="screenshots")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)