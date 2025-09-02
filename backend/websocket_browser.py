import asyncio
import json
import logging
import os
import shutil
import sys
from typing import Dict, Set, Optional
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from fastapi import WebSocket, WebSocketDisconnect
import base64
from pathlib import Path

logger = logging.getLogger(__name__)

def _find_chrome_executable() -> str:
    """Return path to system-installed Chrome/Chromium or '' if not found."""
    if sys.platform.startswith("darwin"):  # macOS
        mac_paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
        for p in mac_paths:
            if os.path.exists(p):
                return p
        return ""

    if sys.platform.startswith("win"):  # Windows
        candidates = [
            os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles%\Chromium\Application\chrome.exe"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return ""

    # Linux
    for name in ("google-chrome-stable", "google-chrome", "chromium-browser", "chromium"):
        p = shutil.which(name)
        if p:
            return p
    return ""

# Global shared browser instance
_shared_browser: Optional[Browser] = None
_shared_playwright = None

async def get_shared_browser() -> Browser:
    """Get or create the shared browser instance"""
    global _shared_browser, _shared_playwright
    
    if _shared_browser and _shared_browser.is_connected():
        return _shared_browser
    
    if _shared_playwright is None:
        _shared_playwright = await async_playwright().start()
    
    # Try system Chrome first
    try:
        _shared_browser = await _shared_playwright.chromium.launch(
            channel="chrome",  # Use system Chrome
            headless=False,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        logger.info("✅ Using system Chrome browser")
        return _shared_browser
    except Exception as e:
        logger.warning(f"System Chrome not found, trying with executable path: {e}")
        
        # Fallback to detected Chrome path
        chrome_path = _find_chrome_executable()
        if chrome_path:
            _shared_browser = await _shared_playwright.chromium.launch(
                executable_path=chrome_path,
                headless=False,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            logger.info(f"✅ Using Chrome from: {chrome_path}")
            return _shared_browser
        else:
            raise RuntimeError(
                "Could not find Google Chrome. Please install Chrome or run 'playwright install' for bundled browser."
            )

class BrowserStreamManager:
    def __init__(self):
        self.browser: Browser = None
        self.context: BrowserContext = None
        self.page: Page = None
        self.active_connections: Set[WebSocket] = set()
        self.current_viewport = {"width": 1920, "height": 1080}
        self._lock = asyncio.Lock()
        self._initialization_in_progress = False
        self._screenshot_task: Optional[asyncio.Task] = None
        self._screenshot_running = False
        
        # Ensure screenshots directory exists
        self.screenshots_dir = Path("screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)

    async def initialize_browser(self):
        """Initialize browser using shared instance"""
        async with self._lock:
            # Prevent multiple initialization attempts
            if self._initialization_in_progress:
                while self._initialization_in_progress:
                    await asyncio.sleep(0.1)
                return
            
            # Reuse if already initialized and connected
            if self.browser and self.browser.is_connected() and self.page:
                return
            
            self._initialization_in_progress = True
            
            try:
                # Use shared browser instance
                self.browser = await get_shared_browser()
                
                self.context = await self.browser.new_context(
                    viewport=self.current_viewport,
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                self.page = await self.context.new_page()
                
                # Enable console logging
                self.page.on("console", lambda msg: logger.info(f"Browser console: {msg.text}"))
                self.page.on("pageerror", lambda exc: logger.error(f"Browser error: {exc}"))
                logger.info("Browser context initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize browser: {e}")
                await self.broadcast_message({
                    "type": "browser_error", 
                    "data": {"message": f"Browser initialization failed: {str(e)}"}
                })
                raise e
            finally:
                self._initialization_in_progress = False

    async def cleanup(self):
        """Clean up browser resources (but keep shared browser alive)"""
        try:
            # Stop screenshot streaming
            if self._screenshot_task and not self._screenshot_task.done():
                self._screenshot_task.cancel()
                try:
                    await self._screenshot_task
                except asyncio.CancelledError:
                    pass
            self._screenshot_running = False
            
            if self.context:
                await self.context.close()
        finally:
            self.context = None
            self.page = None
            self._screenshot_task = None
            # Don't close shared browser - it will be reused

    async def reset_browser(self):
        """Reset browser and notify all connected clients"""
        try:
            await self.cleanup()
            await self.broadcast_message({
                "type": "browser_reset",
                "data": {"message": "Browser session reset successfully"}
            })
            logger.info("Browser session reset successfully")
        except Exception as e:
            logger.error(f"Error resetting browser: {e}")
            await self.broadcast_message({
                "type": "browser_error",
                "data": {"message": f"Browser reset failed: {str(e)}"}
            })

    async def connect_websocket(self, websocket: WebSocket):
        """Add a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        
        try:
            # Initialize browser if not ready
            if not (self.browser and self.browser.is_connected() and self.page):
                await self.initialize_browser()
            
            # Start screenshot streaming if not already running
            if not self._screenshot_running:
                await self.start_screenshot_streaming()
            
            # Send initial state
            await self.send_frame_update()
        except Exception as e:
            logger.exception("Failed during WebSocket connect/initialize.")
            await websocket.send_text(json.dumps({
                "type": "browser_error",
                "data": {"message": f"Browser initialization failed: {str(e)}"}
            }))

    async def disconnect_websocket(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.active_connections.discard(websocket)
        
        # Stop screenshot streaming if no connections left
        if not self.active_connections and self._screenshot_running:
            await self.stop_screenshot_streaming()

    async def broadcast_message(self, message: dict):
        """Send message to all connected clients"""
        if not self.active_connections:
            return
            
        message_str = json.dumps(message)
        disconnected = set()
        
        for websocket in self.active_connections:
            try:
                await websocket.send_text(message_str)
            except Exception as e:
                logger.error(f"Error sending message to websocket: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected websockets
        for ws in disconnected:
            self.active_connections.discard(ws)

    async def get_page_content(self) -> str:
        """Get the current page content as HTML"""
        if not self.page:
            return "<html><body><h1>Browser not initialized</h1></body></html>"
        
        try:
            # Get the full HTML content
            content = await self.page.content()
            
            # Inject base tag to fix relative URLs
            current_url = self.page.url
            base_tag = f'<base href="{current_url}">'
            
            if '<head>' in content:
                content = content.replace('<head>', f'<head>{base_tag}')
            else:
                content = f'{base_tag}{content}'
            
            return content
        except Exception as e:
            logger.error(f"Error getting page content: {e}")
            return f"<html><body><h1>Error loading page: {str(e)}</h1></body></html>"

    async def take_screenshot(self) -> Optional[str]:
        """Take a screenshot and return base64 encoded data"""
        if not self.page:
            return None
            
        try:
            # Take screenshot
            screenshot_bytes = await self.page.screenshot(
                type='jpeg',
                quality=80,
                full_page=False
            )
            
            # Convert to base64
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            
            # Save to file with timestamp
            timestamp = int(asyncio.get_event_loop().time() * 1000)
            filename = f"screenshot_{timestamp}.jpg"
            filepath = self.screenshots_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(screenshot_bytes)
                
            return screenshot_b64
            
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return None

    async def start_screenshot_streaming(self):
        """Start periodic screenshot streaming"""
        if self._screenshot_running:
            return
            
        self._screenshot_running = True
        self._screenshot_task = asyncio.create_task(self._screenshot_loop())
        logger.info("Started screenshot streaming")

    async def stop_screenshot_streaming(self):
        """Stop screenshot streaming"""
        self._screenshot_running = False
        if self._screenshot_task and not self._screenshot_task.done():
            self._screenshot_task.cancel()
            try:
                await self._screenshot_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped screenshot streaming")

    async def _screenshot_loop(self):
        """Periodic screenshot capture loop"""
        while self._screenshot_running and self.active_connections:
            try:
                if self.page:
                    screenshot_b64 = await self.take_screenshot()
                    if screenshot_b64:
                        await self.send_frame_update_with_screenshot(screenshot_b64)
                
                # Wait 1 second before next screenshot
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in screenshot loop: {e}")
                await asyncio.sleep(1.0)

    async def send_frame_update_with_screenshot(self, screenshot_b64: str):
        """Send current page state with screenshot to all connected clients"""
        if not self.page:
            return
            
        try:
            title = await self.page.title()
            url = self.page.url
            
            frame_data = {
                "url": url,
                "title": title,
                "timestamp": asyncio.get_event_loop().time() * 1000,
                "screenshot": screenshot_b64
            }
            
            await self.broadcast_message({
                "type": "browser_frame",
                "data": frame_data
            })
        except Exception as e:
            logger.error(f"Error sending frame update with screenshot: {e}")

    async def send_frame_update(self):
        """Send current page state to all connected clients (URL and title only)"""
        if not self.page:
            return
            
        try:
            title = await self.page.title()
            url = self.page.url
            
            frame_data = {
                "url": url,
                "title": title,
                "timestamp": asyncio.get_event_loop().time() * 1000
            }
            
            await self.broadcast_message({
                "type": "browser_frame",
                "data": frame_data
            })
        except Exception as e:
            logger.error(f"Error sending frame update: {e}")

    async def navigate_to_url(self, url: str, viewport: dict = None):
        """Navigate to a specific URL"""
        async with self._lock:
            try:
                if not (self.browser and self.browser.is_connected() and self.page):
                    await self.initialize_browser()
                
                if viewport and viewport != self.current_viewport:
                    self.current_viewport = viewport
                    await self.page.set_viewport_size(viewport)
                
                # Notify navigation start
                await self.broadcast_message({
                    "type": "navigation_start",
                    "data": {"url": url}
                })
                
                # Navigate to URL
                await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
                
                # Wait a bit for dynamic content
                await asyncio.sleep(1)
                
                # Notify URL change
                await self.broadcast_message({
                    "type": "url_changed",
                    "data": {"url": self.page.url}
                })
                
                # Take screenshot after navigation and send frame update
                screenshot_b64 = await self.take_screenshot()
                if screenshot_b64:
                    await self.send_frame_update_with_screenshot(screenshot_b64)
                else:
                    await self.send_frame_update()
                
            except Exception as e:
                logger.error(f"Error navigating to {url}: {e}")
                await self.broadcast_message({
                    "type": "navigation_error",
                    "data": {"url": url, "error": str(e)}
                })

    async def navigate_back(self):
        """Navigate back in browser history"""
        try:
            if self.page:
                await self.page.go_back(wait_until="domcontentloaded")
                await self.send_frame_update()
        except Exception as e:
            logger.error(f"Error navigating back: {e}")

    async def navigate_forward(self):
        """Navigate forward in browser history"""
        try:
            if self.page:
                await self.page.go_forward(wait_until="domcontentloaded")
                await self.send_frame_update()
        except Exception as e:
            logger.error(f"Error navigating forward: {e}")

    async def handle_websocket_message(self, websocket: WebSocket, message: dict):
        """Handle incoming WebSocket messages"""
        message_type = message.get("type")
        data = message.get("data", {})
        
        try:
            if message_type == "navigate":
                url = data.get("url")
                viewport = data.get("viewport")
                if url:
                    await self.navigate_to_url(url, viewport)
            
            elif message_type == "navigate_back":
                await self.navigate_back()
            
            elif message_type == "navigate_forward":
                await self.navigate_forward()
            
            elif message_type == "init_browser":
                viewport = data.get("viewport")
                if viewport and viewport != self.current_viewport:
                    self.current_viewport = viewport
                    if self.page:
                        await self.page.set_viewport_size(viewport)
                await self.send_frame_update()
            
            elif message_type == "reset_browser":
                await self.reset_browser()
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling websocket message: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"message": str(e)}
            }))

# Global browser manager instance
browser_manager = BrowserStreamManager()

async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for browser streaming"""
    await browser_manager.connect_websocket(websocket)
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await browser_manager.handle_websocket_message(websocket, message)
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "data": {"message": "Invalid JSON message"}
                }))
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
    finally:
        await browser_manager.disconnect_websocket(websocket)