"""
Vision-Based Browser Automation
Uses AI vision models to analyze screenshots and generate precise interactions
"""

import json
import re
import base64
from typing import Dict, List, Tuple, Optional, Any
from playwright.async_api import Page
from ai_providers import ai_manager, AIResponse


class VisionAutomationEngine:
    def __init__(self, page: Page):
        self.page = page
        self.confidence_threshold = 0.7
    
    async def vision_click(self, command: str, screenshot: str = None) -> bool:
        """Use vision AI to find and click elements based on screenshot"""
        
        # Take screenshot if not provided
        if not screenshot:
            screenshot = await self._take_screenshot_b64()
            if not screenshot:
                return False
        
        try:
            # Analyze screenshot with AI
            analysis = await ai_manager.analyze_screenshot_with_fallback(screenshot, command)
            
            # Parse AI response for coordinates
            coordinates = self._parse_ai_coordinates(analysis.content)
            
            if coordinates:
                x, y = coordinates
                print(f"Vision AI found element at coordinates: ({x}, {y})")
                
                # Validate coordinates are within viewport
                viewport = await self.page.viewport_size()
                if 0 <= x <= viewport['width'] and 0 <= y <= viewport['height']:
                    await self.page.click("body", position={"x": x, "y": y})
                    print(f"✅ Vision-based click succeeded at ({x}, {y})")
                    return True
                else:
                    print(f"❌ Coordinates out of viewport: ({x}, {y})")
                    return False
            else:
                print("❌ Could not extract coordinates from AI response")
                return False
                
        except Exception as e:
            print(f"❌ Vision-based click failed: {str(e)}")
            return False
    
    async def vision_fill(self, command: str, value: str, screenshot: str = None) -> bool:
        """Use vision AI to find and fill input fields"""
        
        if not screenshot:
            screenshot = await self._take_screenshot_b64()
            if not screenshot:
                return False
        
        try:
            # Modify command to specify we're looking for input field
            input_command = f"Find input field for: {command}"
            analysis = await ai_manager.analyze_screenshot_with_fallback(screenshot, input_command)
            
            coordinates = self._parse_ai_coordinates(analysis.content)
            
            if coordinates:
                x, y = coordinates
                print(f"Vision AI found input field at: ({x}, {y})")
                
                # Click to focus the input field
                await self.page.click("body", position={"x": x, "y": y})
                await self.page.wait_for_timeout(500)  # Wait for focus
                
                # Clear and fill
                await self.page.keyboard.press("Control+a")  # Select all
                await self.page.keyboard.type(value)
                
                print(f"✅ Vision-based fill succeeded at ({x}, {y})")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Vision-based fill failed: {str(e)}")
            return False
    
    async def vision_verify_action(self, expected_result: str, screenshot_before: str = None) -> bool:
        """Verify if an action had the expected result using vision"""
        
        # Take screenshot after action
        screenshot_after = await self._take_screenshot_b64()
        if not screenshot_after:
            return False
        
        try:
            verify_command = f"Check if this action succeeded: {expected_result}"
            analysis = await ai_manager.analyze_screenshot_with_fallback(screenshot_after, verify_command)
            
            # Simple confidence check
            confidence = self._extract_confidence(analysis.content)
            return confidence > self.confidence_threshold
            
        except Exception as e:
            print(f"❌ Vision verification failed: {str(e)}")
            return False
    
    async def vision_analyze_page(self, screenshot: str = None) -> Dict[str, Any]:
        """Analyze page structure and identify interactive elements"""
        
        if not screenshot:
            screenshot = await self._take_screenshot_b64()
            if not screenshot:
                return {}
        
        try:
            analyze_command = "Identify all clickable elements, input fields, and buttons on this page. Return JSON format with element types and approximate coordinates."
            analysis = await ai_manager.analyze_screenshot_with_fallback(screenshot, analyze_command)
            
            # Try to parse as JSON
            try:
                return json.loads(analysis.content)
            except:
                # If not JSON, return basic structure
                return {
                    "analysis": analysis.content,
                    "provider": analysis.provider,
                    "confidence": analysis.confidence
                }
                
        except Exception as e:
            print(f"❌ Vision page analysis failed: {str(e)}")
            return {}
    
    async def vision_enhanced_selector_generation(self, target_text: str, screenshot: str = None) -> List[str]:
        """Use vision to generate better selectors based on visual context"""
        
        if not screenshot:
            screenshot = await self._take_screenshot_b64()
            if not screenshot:
                return []
        
        try:
            command = f"Generate CSS selectors for element containing '{target_text}'. Consider visual layout, nearby elements, and element hierarchy."
            analysis = await ai_manager.analyze_screenshot_with_fallback(screenshot, command)
            
            # Extract selectors from AI response
            selectors = self._extract_selectors_from_response(analysis.content)
            return selectors
            
        except Exception as e:
            print(f"❌ Vision selector generation failed: {str(e)}")
            return []
    
    def _parse_ai_coordinates(self, ai_response: str) -> Optional[Tuple[int, int]]:
        """Parse coordinates from AI response"""
        try:
            # Try to parse JSON first
            if ai_response.strip().startswith('{'):
                data = json.loads(ai_response)
                if 'x' in data and 'y' in data:
                    return (int(data['x']), int(data['y']))
                if 'coordinates' in data:
                    coords = data['coordinates']
                    return (int(coords['x']), int(coords['y']))
            
            # Try regex patterns
            patterns = [
                r'"x":\s*(\d+),\s*"y":\s*(\d+)',
                r'x:\s*(\d+),\s*y:\s*(\d+)',
                r'\((\d+),\s*(\d+)\)',
                r'coordinates.*?(\d+).*?(\d+)',
                r'position.*?(\d+).*?(\d+)',
                r'click.*?(\d+).*?(\d+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, ai_response, re.IGNORECASE)
                if match:
                    return (int(match.group(1)), int(match.group(2)))
            
            return None
            
        except Exception as e:
            print(f"Error parsing coordinates: {e}")
            return None
    
    def _extract_confidence(self, ai_response: str) -> float:
        """Extract confidence score from AI response"""
        try:
            # Look for confidence patterns
            patterns = [
                r'"confidence":\s*([\d.]+)',
                r'confidence.*?([\d.]+)',
                r'success.*?([\d.]+)',
                r'probability.*?([\d.]+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, ai_response, re.IGNORECASE)
                if match:
                    return float(match.group(1))
            
            # Default confidence based on keywords
            positive_keywords = ['success', 'found', 'located', 'identified', 'clicked']
            negative_keywords = ['failed', 'not found', 'error', 'unable', 'cannot']
            
            response_lower = ai_response.lower()
            positive_count = sum(1 for word in positive_keywords if word in response_lower)
            negative_count = sum(1 for word in negative_keywords if word in response_lower)
            
            if positive_count > negative_count:
                return 0.8
            elif negative_count > positive_count:
                return 0.2
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _extract_selectors_from_response(self, ai_response: str) -> List[str]:
        """Extract CSS selectors from AI response"""
        selectors = []
        
        try:
            # Try to parse JSON
            if ai_response.strip().startswith('{') or ai_response.strip().startswith('['):
                data = json.loads(ai_response)
                if isinstance(data, list):
                    selectors.extend([s for s in data if isinstance(s, str)])
                elif isinstance(data, dict) and 'selectors' in data:
                    selectors.extend(data['selectors'])
        except:
            pass
        
        # Extract from text using patterns
        patterns = [
            r'[\'"]([.#][a-zA-Z][a-zA-Z0-9_-]*)[\'"]',  # CSS classes and IDs
            r'[\'"]([a-zA-Z]+\[[^\]]+\])[\'"]',  # Attribute selectors
            r'[\'"]([a-zA-Z]+:[a-zA-Z-]+\([^)]*\))[\'"]',  # Pseudo selectors
            r'selector:\s*[\'"]([^\'\"]+)[\'"]',  # Explicit selector fields
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, ai_response)
            selectors.extend(matches)
        
        # Clean and validate selectors
        valid_selectors = []
        for selector in selectors:
            if selector and len(selector) > 1 and not selector.isspace():
                valid_selectors.append(selector.strip())
        
        return list(set(valid_selectors))  # Remove duplicates
    
    async def _take_screenshot_b64(self) -> Optional[str]:
        """Take screenshot and return base64 encoded string"""
        try:
            screenshot_bytes = await self.page.screenshot(
                type='jpeg',
                quality=80,
                full_page=False
            )
            return base64.b64encode(screenshot_bytes).decode('utf-8')
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            return None


class VisionCommandGenerator:
    def __init__(self, page: Page):
        self.page = page
        self.vision_engine = VisionAutomationEngine(page)
    
    async def generate_vision_enhanced_code(self, command: str, dom: str = "") -> str:
        """Generate Playwright code enhanced with vision capabilities"""
        
        # Take screenshot for analysis
        screenshot = await self.vision_engine._take_screenshot_b64()
        
        try:
            # Get AI to generate code with vision context
            vision_command = f"Generate Playwright code for: {command}. Use vision analysis if needed for better element targeting."
            
            result = await ai_manager.generate_code_with_fallback(
                vision_command, 
                dom, 
                screenshot
            )
            
            # Enhance code with vision fallbacks
            enhanced_code = self._add_vision_fallbacks(result.content, command)
            
            return enhanced_code
            
        except Exception as e:
            print(f"❌ Vision-enhanced code generation failed: {str(e)}")
            # Fallback to regular code generation
            return await ai_manager.generate_code_with_fallback(command, dom)
    
    def _add_vision_fallbacks(self, original_code: str, command: str) -> str:
        """Add vision-based fallback strategies to generated code"""
        
        # Extract click operations
        click_pattern = r'await page\.click\([\'"]([^\'"]+)[\'"]\)'
        clicks = re.findall(click_pattern, original_code)
        
        enhanced_code = original_code
        
        # Add try-catch with vision fallback for each click
        for selector in clicks:
            original_click = f"await page.click('{selector}')"
            
            enhanced_click = f"""
try:
    {original_click}
except Exception as e:
    print(f"Regular click failed: {{e}}, trying vision-based approach...")
    # Vision-based fallback
    from vision_automation import VisionAutomationEngine
    vision_engine = VisionAutomationEngine(page)
    success = await vision_engine.vision_click("{command}")
    if not success:
        raise Exception(f"Both regular and vision-based clicks failed for: {selector}")
"""
            enhanced_code = enhanced_code.replace(original_click, enhanced_click.strip())
        
        return enhanced_code