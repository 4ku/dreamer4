#!/usr/bin/env python3
"""Capture screenshots of the Dreamer 4 World Model Explorer"""
import time
import sys
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from pyvirtualdisplay import Display

def capture_screenshots(url="http://localhost:8765/", wait_time=3):
    """Capture screenshots of the web UI"""
    
    # Start virtual display
    print("Starting virtual display...")
    display = Display(visible=0, size=(1400, 1000))
    display.start()
    
    try:
        # Setup Firefox
        print("Setting up Firefox...")
        options = Options()
        options.add_argument('--width=1400')
        options.add_argument('--height=1000')
        
        driver = webdriver.Firefox(options=options)
        driver.set_window_size(1400, 1000)
        
        try:
            # Take initial screenshot
            print(f"Navigating to {url}...")
            driver.get(url)
            time.sleep(1)
            
            initial_path = "dreamer4_web_initial.png"
            print(f"Taking initial screenshot: {initial_path}")
            driver.save_screenshot(initial_path)
            print(f"✓ Saved {initial_path}")
            
            # Get initial status
            try:
                status = driver.find_element(By.CLASS_NAME, "status").text
                print(f"Initial status: {status}")
            except:
                print("Could not read initial status")
            
            # Wait for connection
            print(f"\nWaiting {wait_time} seconds for WebSocket connection...")
            time.sleep(wait_time)
            
            # Take second screenshot
            connected_path = "dreamer4_web_connected.png"
            print(f"Taking connected screenshot: {connected_path}")
            driver.save_screenshot(connected_path)
            print(f"✓ Saved {connected_path}")
            
            # Get page info
            print("\n=== Page Status ===")
            try:
                status = driver.find_element(By.CLASS_NAME, "status").text
                print(f"Status: {status}")
            except Exception as e:
                print(f"Could not read status: {e}")
            
            # Get metrics
            print("\n=== Metrics ===")
            try:
                metrics = driver.find_elements(By.CLASS_NAME, "metric")
                for metric in metrics[:6]:
                    try:
                        label = metric.find_element(By.CLASS_NAME, "label").text
                        value = metric.find_element(By.CLASS_NAME, "value").text
                        print(f"{label}: {value}")
                    except:
                        pass
            except Exception as e:
                print(f"Could not read metrics: {e}")
            
            # Check for images
            print("\n=== Images ===")
            try:
                env_img = driver.find_element(By.CSS_SELECTOR, ".panel:nth-child(1) img")
                wm_img = driver.find_element(By.CSS_SELECTOR, ".panel:nth-child(2) img")
                print(f"Environment image src length: {len(env_img.get_attribute('src'))}")
                print(f"World Model image src length: {len(wm_img.get_attribute('src'))}")
            except Exception as e:
                print(f"Could not check images: {e}")
            
            print("\n✓ Screenshot capture completed successfully")
            return True
            
        finally:
            driver.quit()
            
    finally:
        display.stop()

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765/"
    wait = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    capture_screenshots(url, wait)
