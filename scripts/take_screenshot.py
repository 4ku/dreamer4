#!/usr/bin/env python3
"""Take screenshots of the Dreamer 4 World Model Explorer"""
import sys
import time
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def take_screenshot(url, output_path, wait_seconds=0):
    """Take a screenshot of a URL"""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--width=1400')
    options.add_argument('--height=1000')
    options.binary_location = '/snap/bin/firefox'
    
    service = Service(executable_path='/snap/bin/geckodriver')
    driver = webdriver.Firefox(service=service, options=options)
    
    try:
        print(f"Navigating to {url}...")
        driver.get(url)
        
        if wait_seconds > 0:
            print(f"Waiting {wait_seconds} seconds...")
            time.sleep(wait_seconds)
        
        # Get page info
        title = driver.title
        print(f"Page title: {title}")
        
        # Try to get connection status
        try:
            status_elem = driver.find_element(By.CLASS_NAME, "status")
            status_text = status_elem.text
            print(f"Status: {status_text}")
        except:
            print("Status element not found")
        
        # Try to get metrics
        try:
            metrics = driver.find_elements(By.CLASS_NAME, "metric")
            print(f"Found {len(metrics)} metrics")
            for metric in metrics[:6]:  # First 6 metrics
                try:
                    label = metric.find_element(By.CLASS_NAME, "label").text
                    value = metric.find_element(By.CLASS_NAME, "value").text
                    print(f"  {label}: {value}")
                except:
                    pass
        except Exception as e:
            print(f"Could not read metrics: {e}")
        
        print(f"Taking screenshot: {output_path}")
        driver.save_screenshot(output_path)
        print(f"Screenshot saved successfully")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        driver.quit()

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765/"
    output = sys.argv[2] if len(sys.argv) > 2 else "screenshot.png"
    wait = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    
    take_screenshot(url, output, wait)
