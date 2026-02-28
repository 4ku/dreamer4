#!/usr/bin/env python3
"""
Simple script to capture a screenshot of localhost:8765 using Firefox headless mode.
"""
import subprocess
import time
import os

def capture_screenshot():
    """Capture screenshot using Firefox in headless mode with geckodriver."""
    try:
        from selenium import webdriver
        from selenium.webdriver.firefox.options import Options
        from selenium.webdriver.firefox.service import Service
        
        # Setup Firefox options
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--width=1280')
        options.add_argument('--height=960')
        options.binary_location = '/snap/firefox/current/usr/lib/firefox/firefox'
        
        # Create driver
        driver = webdriver.Firefox(options=options)
        
        try:
            # Navigate to the page
            print("Navigating to http://localhost:8765/...")
            driver.get('http://localhost:8765/')
            
            # Wait for page to load
            time.sleep(2)
            
            # Take screenshot
            screenshot_path = '/mnt/dsk1/dreamer4/dreamer4_explorer_screenshot.png'
            driver.save_screenshot(screenshot_path)
            print(f"Screenshot saved to: {screenshot_path}")
            
            # Get page title
            title = driver.title
            print(f"Page title: {title}")
            
            # Get page source to verify elements
            page_source = driver.page_source
            
            # Check for key elements
            checks = {
                'Title contains "Dreamer 4"': 'Dreamer 4' in title,
                'Environment panel': 'Environment' in page_source,
                'World Model panel': 'World Model' in page_source,
                'Left/Right buttons': '&larr;' in page_source and '&rarr;' in page_source,
                'Policy button': 'Policy' in page_source,
                'Reset button': 'Reset' in page_source,
                'Mode metric': 'Mode' in page_source,
                'Step metric': 'Step' in page_source,
                'Return metric': 'Return' in page_source,
                'Reward metric': 'Reward' in page_source,
                'Pred R metric': 'Pred R' in page_source,
                'Pred V metric': 'Pred V' in page_source,
            }
            
            print("\nPage verification:")
            for check, result in checks.items():
                status = "✓" if result else "✗"
                print(f"  {status} {check}")
            
            return True
            
        finally:
            driver.quit()
            
    except ImportError:
        print("Selenium not installed. Installing...")
        subprocess.run(['pip', 'install', 'selenium'], check=True)
        print("Please run the script again.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == '__main__':
    capture_screenshot()
