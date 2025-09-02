import asyncio
import json
import logging
from typing import Dict, Set
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from fastapi import WebSocket, WebSocketDisconnect
import base64

logger = logging.getLogger(__name__)

class BrowserStreamManager:
    def __init__(self):
        self.playwright = None
        self.browser: Browser = None
        self.context: BrowserContext = None
        self.page: Page = None
        self.active_connections: Set[WebSocket] = set()
        self.current_viewport = {"width": 1920, "height": 1080}
        self._lock = asyncio.Lock()

    async def initialize_browser(self):
        """Initialize Playwright browser instance"""
        try:
            if self.playwright is None:
                self.playwright = await async_playwright().start()
                self.browser = await self.playwright.chromium.launch(
                    headless=False,  # Set to True for headless mode
                    args=['--no-sandbox', '--disable-setuid-sandbox']
                )
                self.context = await self.browser.new_context(
                    viewport=self.current_viewport,
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                self.page = await self.context.new_page()
                
                # Enable console logging
                self.page.on("console", lambda msg: logger.info(f"Browser console: {msg.text}"))
                self.page.on("pageerror", lambda exc: logger.error(f"Browser error: {exc}"))
                logger.info("Browser initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            logger.error("Make sure to run 'playwright install' in the backend directory first")
            # Send error to connected clients
            await self.broadcast_message({
                "type": "browser_error",
                "data": {"message": f"Browser initialization failed: {str(e)}"}
            })
            raise e

    async def cleanup(self):
        """Clean up browser resources"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def connect_websocket(self, websocket: WebSocket):
        """Add a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        
        # Initialize browser if not already done
        if self.browser is None:
            await self.initialize_browser()
        
        # Send initial state
        await self.send_frame_update()

    async def disconnect_websocket(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.active_connections.discard(websocket)

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

    async def send_frame_update(self):
        """Send current page state to all connected clients"""
        if not self.page:
            return
            
        try:
            content = await self.get_page_content()
            title = await self.page.title()
            url = self.page.url
            
            frame_data = {
                "url": url,
                "title": title,
                "content": content,
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
                if not self.page:
                    await self.initialize_browser()
                
                if viewport and viewport != self.current_viewport:
                    self.current_viewport = viewport
                    await self.page.set_viewport_size(viewport["width"], viewport["height"])
                
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
                
                # Send updated frame
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
                        await self.page.set_viewport_size(viewport["width"], viewport["height"])
                await self.send_frame_update()
            
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