#!/usr/bin/env python3
"""
Clean up the dashboard file by removing all test methods and replacing them with simple stubs.
This will prevent SSE connection conflicts from test code.
"""

import re

def cleanup_dashboard():
    """Clean up the dashboard file by removing test methods."""
    
    # Read the dashboard file
    with open('gan_dashboard.py', 'r') as f:
        content = f.read()
    
    # Define patterns to find and replace test methods
    test_methods = [
        r'# async def test_minimal_sse\(self, request\):.*?return web\.Response\(text=html, content_type="text/html"\)',
        r'# async def test_main_dashboard_sse\(self, request\):.*?return web\.Response\(text=html, content_type="text/html"\)',
        r'# async def test_ui_updates\(self, request\):.*?return web\.Response\(text=html, content_type="text/html"\)',
        r'# async def debug_sse_connection\(self, request\):.*?return web\.Response\(text=html_content, content_type="text/html"\)',
        r'# async def test_minimal_sse\(self, request\):.*?return web\.Response\(text=html_content, content_type="text/html"\)'
    ]
    
    # Replace each test method with a simple stub
    for pattern in test_methods:
        content = re.sub(pattern, 
                        '# Test method removed to prevent SSE connection conflicts\n        pass',
                        content, flags=re.DOTALL)
    
    # Write the cleaned content back
    with open('gan_dashboard.py', 'w') as f:
        f.write(content)
    
    print("✅ Dashboard file cleaned up - test methods removed")

if __name__ == "__main__":
    cleanup_dashboard() 