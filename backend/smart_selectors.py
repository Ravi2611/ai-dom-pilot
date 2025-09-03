"""
Smart Selector Generation and Retry System
Implements automatic alternative locator generation with fuzzy matching
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from playwright.async_api import Page, ElementHandle
from difflib import SequenceMatcher
from bs4 import BeautifulSoup
import asyncio


class SelectorStrategy:
    def __init__(self, name: str, selector: str, confidence: float):
        self.name = name
        self.selector = selector
        self.confidence = confidence


class SmartSelectorGenerator:
    def __init__(self):
        self.selector_patterns = {
            'exact_text': lambda text: f"text='{text}'",
            'partial_text': lambda text: f"text*='{text}'", 
            'case_insensitive': lambda text: f"text=/{re.escape(text)}/i",
            'contains_text': lambda text: f":has-text('{text}')",
            'css_class': lambda class_name: f".{class_name}",
            'id': lambda id_val: f"#{id_val}",
            'name_attr': lambda name: f"[name='{name}']",
            'placeholder': lambda text: f"[placeholder*='{text}' i]",
            'aria_label': lambda label: f"[aria-label*='{label}' i]",
            'role': lambda role: f"[role='{role}']",
            'data_attr': lambda key, val: f"[data-{key}*='{val}' i]"
        }
    
    async def generate_selectors_for_text(self, page: Page, target_text: str, dom: str = "") -> List[SelectorStrategy]:
        """Generate multiple selector strategies for finding text-based elements"""
        strategies = []
        
        # Clean and normalize text
        clean_text = target_text.strip()
        words = clean_text.split()
        
        # Strategy 1: Exact text match
        strategies.append(SelectorStrategy(
            "exact_text",
            self.selector_patterns['exact_text'](clean_text),
            0.9
        ))
        
        # Strategy 2: Partial text match
        if len(words) > 1:
            for i in range(len(words)):
                partial = ' '.join(words[i:min(i+3, len(words))])  # Take 1-3 words
                strategies.append(SelectorStrategy(
                    f"partial_text_{i}",
                    self.selector_patterns['partial_text'](partial),
                    0.7 - (i * 0.1)
                ))
        
        # Strategy 3: Case insensitive
        strategies.append(SelectorStrategy(
            "case_insensitive",
            self.selector_patterns['case_insensitive'](clean_text),
            0.8
        ))
        
        # Strategy 4: Contains text (for containers)
        strategies.append(SelectorStrategy(
            "contains_text",
            self.selector_patterns['contains_text'](clean_text),
            0.6
        ))
        
        # Strategy 5: Button/link with text
        for tag in ['button', 'a', 'input[type="button"]', 'input[type="submit"]']:
            strategies.append(SelectorStrategy(
                f"{tag}_with_text",
                f"{tag}:has-text('{clean_text}')",
                0.8
            ))
        
        # Strategy 6: Parse DOM for better selectors
        if dom:
            dom_strategies = await self._analyze_dom_for_selectors(clean_text, dom)
            strategies.extend(dom_strategies)
        
        # Sort by confidence
        strategies.sort(key=lambda x: x.confidence, reverse=True)
        return strategies[:10]  # Return top 10 strategies
    
    async def _analyze_dom_for_selectors(self, target_text: str, dom: str) -> List[SelectorStrategy]:
        """Analyze DOM to find more specific selectors"""
        strategies = []
        
        try:
            soup = BeautifulSoup(dom, 'html.parser')
            
            # Find elements containing the text
            elements = soup.find_all(string=re.compile(re.escape(target_text), re.IGNORECASE))
            
            for text_node in elements[:5]:  # Limit to first 5 matches
                element = text_node.parent
                if not element:
                    continue
                
                # Generate CSS selector based on attributes
                css_parts = [element.name]
                
                # Add ID if present
                if element.get('id'):
                    strategies.append(SelectorStrategy(
                        f"id_{element.get('id')}",
                        f"#{element.get('id')}",
                        0.95
                    ))
                
                # Add class-based selectors
                if element.get('class'):
                    classes = element.get('class')
                    if isinstance(classes, list):
                        for cls in classes:
                            if cls and not cls.startswith('_'):  # Skip generated classes
                                strategies.append(SelectorStrategy(
                                    f"class_{cls}",
                                    f".{cls}",
                                    0.7
                                ))
                
                # Add attribute-based selectors
                for attr in ['name', 'value', 'aria-label', 'title']:
                    if element.get(attr):
                        strategies.append(SelectorStrategy(
                            f"attr_{attr}",
                            f"[{attr}='{element.get(attr)}']",
                            0.6
                        ))
                
                # Add data attribute selectors
                for attr, value in element.attrs.items():
                    if attr.startswith('data-') and isinstance(value, str):
                        strategies.append(SelectorStrategy(
                            f"data_attr_{attr}",
                            f"[{attr}='{value}']",
                            0.5
                        ))
        
        except Exception as e:
            print(f"Error analyzing DOM: {e}")
        
        return strategies
    
    async def generate_coordinate_selector(self, x: int, y: int) -> SelectorStrategy:
        """Generate coordinate-based click strategy"""
        return SelectorStrategy(
            "coordinates",
            f"coordinate_click({x}, {y})",
            0.3  # Lower confidence as it's less reliable
        )
    
    def fuzzy_match_elements(self, target_text: str, dom: str, threshold: float = 0.6) -> List[SelectorStrategy]:
        """Find elements with fuzzy text matching"""
        strategies = []
        
        try:
            soup = BeautifulSoup(dom, 'html.parser')
            
            # Get all text elements
            text_elements = []
            for element in soup.find_all(text=True):
                if element.parent and element.strip():
                    text_elements.append((element.strip(), element.parent))
            
            # Find fuzzy matches
            for text, element in text_elements:
                similarity = SequenceMatcher(None, target_text.lower(), text.lower()).ratio()
                
                if similarity >= threshold:
                    # Create selector for fuzzy match
                    if element.get('id'):
                        selector = f"#{element.get('id')}"
                    elif element.get('class'):
                        classes = element.get('class')
                        if isinstance(classes, list):
                            selector = f".{classes[0]}"
                        else:
                            selector = f".{classes}"
                    else:
                        selector = f"text=/{re.escape(text)}/i"
                    
                    strategies.append(SelectorStrategy(
                        f"fuzzy_match_{similarity:.2f}",
                        selector,
                        similarity * 0.6  # Reduce confidence for fuzzy matches
                    ))
        
        except Exception as e:
            print(f"Error in fuzzy matching: {e}")
        
        return sorted(strategies, key=lambda x: x.confidence, reverse=True)


class SmartRetrySystem:
    def __init__(self, page: Page):
        self.page = page
        self.selector_generator = SmartSelectorGenerator()
        self.max_retries = 5
        self.retry_delay = 1.0
    
    async def smart_click(self, target_text: str, dom: str = "", screenshot: str = "") -> bool:
        """Smart click with multiple retry strategies"""
        
        # Generate selector strategies
        strategies = await self.selector_generator.generate_selectors_for_text(
            self.page, target_text, dom
        )
        
        # Add fuzzy matching strategies
        if dom:
            fuzzy_strategies = self.selector_generator.fuzzy_match_elements(target_text, dom)
            strategies.extend(fuzzy_strategies)
        
        # Try each strategy
        for i, strategy in enumerate(strategies):
            try:
                print(f"Attempt {i+1}: Trying {strategy.name} - {strategy.selector}")
                
                if strategy.selector.startswith("coordinate_click"):
                    # Handle coordinate-based click
                    coords = self._parse_coordinates(strategy.selector)
                    if coords:
                        await self.page.click(f"body", position={"x": coords[0], "y": coords[1]})
                        print(f"✅ Coordinate click succeeded at {coords}")
                        return True
                else:
                    # Handle regular selectors
                    element = await self.page.wait_for_selector(strategy.selector, timeout=3000)
                    if element:
                        # Check if element is visible and enabled
                        is_visible = await element.is_visible()
                        is_enabled = await element.is_enabled()
                        
                        if is_visible and is_enabled:
                            await element.click()
                            print(f"✅ Click succeeded with {strategy.name}")
                            return True
                        else:
                            print(f"⚠️ Element found but not clickable (visible: {is_visible}, enabled: {is_enabled})")
            
            except Exception as e:
                print(f"❌ Strategy {strategy.name} failed: {str(e)}")
                continue
        
        print(f"❌ All click strategies failed for: {target_text}")
        return False
    
    async def smart_fill(self, target_text: str, value: str, dom: str = "") -> bool:
        """Smart fill with multiple retry strategies"""
        
        # Generate input-specific strategies
        strategies = []
        
        # Basic input selectors
        input_patterns = [
            f"input:has-text('{target_text}')",
            f"input[placeholder*='{target_text}' i]",
            f"input[name*='{target_text.lower()}']",
            f"input[id*='{target_text.lower()}']",
            f"textarea[placeholder*='{target_text}' i]",
            f"[contenteditable]:has-text('{target_text}')"
        ]
        
        for i, pattern in enumerate(input_patterns):
            strategies.append(SelectorStrategy(
                f"input_pattern_{i}",
                pattern,
                0.8 - (i * 0.1)
            ))
        
        # Try each strategy
        for strategy in strategies:
            try:
                print(f"Trying fill strategy: {strategy.name} - {strategy.selector}")
                
                element = await self.page.wait_for_selector(strategy.selector, timeout=3000)
                if element:
                    is_visible = await element.is_visible()
                    is_enabled = await element.is_enabled()
                    
                    if is_visible and is_enabled:
                        await element.fill(value)
                        print(f"✅ Fill succeeded with {strategy.name}")
                        return True
            
            except Exception as e:
                print(f"❌ Fill strategy {strategy.name} failed: {str(e)}")
                continue
        
        print(f"❌ All fill strategies failed for: {target_text}")
        return False
    
    def _parse_coordinates(self, coordinate_selector: str) -> Optional[Tuple[int, int]]:
        """Parse coordinates from coordinate_click(x, y) format"""
        match = re.match(r"coordinate_click\((\d+),\s*(\d+)\)", coordinate_selector)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return None
    
    async def wait_for_element_intelligently(self, selector: str, timeout: int = 30000) -> Optional[ElementHandle]:
        """Wait for element with intelligent retry strategies"""
        
        # Basic wait
        try:
            return await self.page.wait_for_selector(selector, timeout=timeout//3)
        except:
            pass
        
        # Wait for network idle (for dynamic content)
        try:
            await self.page.wait_for_load_state('networkidle', timeout=timeout//3)
            return await self.page.wait_for_selector(selector, timeout=timeout//3)
        except:
            pass
        
        # Wait for DOM content loaded
        try:
            await self.page.wait_for_load_state('domcontentloaded', timeout=timeout//3)
            return await self.page.wait_for_selector(selector, timeout=timeout//3)
        except:
            pass
        
        return None