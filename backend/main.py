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
from bs4 import BeautifulSoup, Comment
from websocket_browser import websocket_endpoint, browser_manager

app = FastAPI(title="AI Browser Automation API")

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

# Global browser instance
browser = None
page = None

def shrink_dom(dom: str) -> str:
    """Compress DOM by keeping only actionable elements"""
    if not dom:
        return ""
    
    soup = BeautifulSoup(dom, "html.parser")
    
    # Remove scripts, styles, comments
    for tag in soup(["script", "style"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    
    # Keep only actionable elements
    form_elements = soup.find_all(["form", "input", "button", "a", "select", "textarea", "div", "span"])
    return "\n".join(str(el) for el in form_elements[:50])  # Limit to first 50 elements

def ai_to_code(command: str, dom: str = "") -> str:
    """Convert natural language + DOM into runnable Playwright code"""
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
    
    # Generic fallback to LLM
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that converts natural language automation commands "
                "into direct runnable Python code using Playwright async API. "
                "The code will be executed inside a context where a `page` object exists. "
                "Use await for all Playwright methods. Do NOT define functions, classes, imports, or variables. "
                "Use DOM to infer selectors if possible. Return only raw code, no markdown fences. "
                "Always use await with page methods like page.click(), page.fill(), page.goto(), etc."
            ),
        },
        {"role": "user", "content": f"Command: {command}\n\nDOM:\n{dom}" if dom else command},
    ]
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0,
        )
        
        code = response.choices[0].message.content.strip()
        
        # Clean accidental markdown fences
        if code.startswith("```"):
            code = code.split("```")[1]
            if code.startswith("python"):
                code = code[len("python"):].strip()
            code = code.split("```")[0].strip()
        
        return code
    except Exception as e:
        return f'# Error generating code: {str(e)}\nawait page.wait_for_timeout(1000)'

async def take_screenshot(command_id: int) -> str:
    """Take screenshot and save it"""
    try:
        os.makedirs("screenshots", exist_ok=True)
        screenshot_path = f"screenshots/step_{command_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        await page.screenshot(path=screenshot_path)
        return screenshot_path
    except Exception as e:
        print(f"Screenshot error: {e}")
        return ""

async def execute_automation_command(command_id: int, command: str, generated_code: str):
    """Execute the automation command in background"""
    global page, browser
    
    conn = sqlite3.connect('automation.db')
    
    try:
        # Initialize browser if not exists
        if not browser or not page:
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(headless=False)
            page = await browser.new_page()
        
        # Update status to running
        conn.execute("UPDATE commands SET status = 'running' WHERE id = ?", (command_id,))
        conn.commit()
        
        # Execute the generated code
        local_vars = {"page": page}
        exec_code = compile(generated_code, "<string>", "exec")
        await eval(exec_code, {"__builtins__": __builtins__, "page": page}, local_vars)
        
        # Take screenshot
        screenshot_path = await take_screenshot(command_id)
        
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
        if page:
            try:
                current_dom = await page.content()
                current_dom = shrink_dom(current_dom)
            except:
                pass
        
        # Use provided DOM or current DOM
        dom_to_use = command.dom if command.dom else current_dom
        
        # Generate code
        generated_code = ai_to_code(command.command, dom_to_use)
        
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
    if page:
        try:
            return {"url": page.url}
        except:
            return {"url": "about:blank"}
    return {"url": "about:blank"}

@app.get("/api/automation/browser/dom")
async def get_current_dom():
    """Get current page DOM (compressed)"""
    if page:
        try:
            dom = await page.content()
            return {"dom": shrink_dom(dom)}
        except Exception as e:
            return {"dom": "", "error": str(e)}
    return {"dom": ""}

# WebSocket endpoint for browser streaming
@app.websocket("/ws/browser")
async def websocket_browser_endpoint(websocket: WebSocket):
    await websocket_endpoint(websocket)

# Serve screenshots
app.mount("/screenshots", StaticFiles(directory="screenshots"), name="screenshots")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up browser on shutdown"""
    global browser
    await browser_manager.cleanup()
    if browser:
        await browser.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)