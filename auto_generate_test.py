import time
import json
import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, WebDriverException

# Try to get the public URL from ngrok API or tunnel_info.json, fallback to localhost
def get_swagger_url():
    # First try to get from tunnel_info.json
    try:
        if os.path.exists("tunnel_info.json"):
            with open("tunnel_info.json", "r") as f:
                tunnel_info = json.load(f)
                return tunnel_info.get("swagger_url", "http://localhost:8000/docs")
    except Exception as e:
        print(f"⚠️  Error reading tunnel_info.json: {e}")
    
    # Try to get from ngrok API
    try:
        response = requests.get("http://localhost:4040/api/tunnels", timeout=2)
        if response.status_code == 200:
            tunnels = response.json()
            for tunnel in tunnels.get("tunnels", []):
                if tunnel.get("config", {}).get("addr") == "http://127.0.0.1:8000":
                    public_url = tunnel.get("public_url")
                    if public_url:
                        swagger_url = f"{public_url}/docs"
                        print(f"✅ Found ngrok tunnel: {public_url}")
                        return swagger_url
    except Exception as e:
        print(f"⚠️  Could not connect to ngrok API: {e}")
    
    # Manual input fallback
    print("⚠️  No ngrok tunnel detected automatically")
    print("💡 Make sure to:")
    print("   1. Start the FastAPI server: python -m app.main")
    print("   2. Start ngrok tunnel: python ngrok.py")
    print("   3. Copy the public URL and add /docs to it")
    
    manual_url = input("🌐 Enter the ngrok public URL (e.g., https://abc123.ngrok.io): ").strip()
    if manual_url:
        if not manual_url.startswith("http"):
            manual_url = "https://" + manual_url
        if not manual_url.endswith("/docs"):
            manual_url = manual_url.rstrip("/") + "/docs"
        return manual_url
    
    print("⚠️  Using localhost as fallback")
    return "http://localhost:8000/docs"

def check_server_running(url):
    """Check if the FastAPI server is running"""
    try:
        # Remove /docs from URL to check the base endpoint
        base_url = url.replace("/docs", "")
        response = requests.get(f"{base_url}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ FastAPI server is running")
            return True
        else:
            print(f"⚠️  Server responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Could not connect to server: {e}")
        return False

SWAGGER_URL = get_swagger_url()
CHROMEDRIVER_PATH = "C:/Users/HP/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe"

print(f"🎯 Using Swagger URL: {SWAGGER_URL}")

prompts = [
    "car",
    "bus",
    "train",
    "plane",
    "give a 5 line python code to print hello world",
    "give a 5 line java code to print hello world",
    "give a 5 line c code to print hello world",
    "give a 5 line c++ code to print hello world",
    "priority",
    "lower"
]
priority = "high"

def find_swagger_endpoint(driver):
    try:
        post_operations = driver.find_elements(By.CSS_SELECTOR, ".opblock-summary-post")
        for op in post_operations:
            if "/generate/" in op.text:
                op.click()
                time.sleep(1)
                section = op.find_element(By.XPATH, "./ancestor::div[contains(@class, 'opblock')]")
                return section
    except Exception:
        pass
    try:
        section = driver.find_element(By.CSS_SELECTOR, "[id*='generate']")
        return section
    except Exception:
        pass
    return None

def find_request_body_input(section):
    try:
        return section.find_element(By.TAG_NAME, "textarea")
    except NoSuchElementException:
        try:
            return section.find_element(By.CSS_SELECTOR, 'div[contenteditable="true"]')
        except NoSuchElementException:
            return None
#what if there is no textarea?
def clear_and_fill_textarea(textarea, new_text):
    try:
        textarea.click()
        time.sleep(0.1)
        try:
            textarea.clear()
        except Exception:
            textarea.send_keys(Keys.CONTROL + 'a')
            textarea.send_keys(Keys.DELETE)
        time.sleep(0.2)
        textarea.send_keys(new_text)
        time.sleep(0.3)
    except Exception as e:
        print(f"Error clearing/filling textarea: {e}")

def find_execute_button(section):
    try:
        return section.find_element(By.XPATH, ".//button[contains(text(), 'Execute')]")
    except NoSuchElementException:
        return None

def find_status_code(section):
    status_selectors = [
        ".responses-inner .response .response_code",
        ".response-col_status",
        ".responses-inner .response_status"
    ]
    for selector in status_selectors:
        try:
            status_elem = section.find_element(By.CSS_SELECTOR, selector)
            code = status_elem.text.strip()
            if code:
                return code
        except NoSuchElementException:
            continue
    return ""

def wait_for_new_status_code(section, prev_code, timeout=60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        code = find_status_code(section)
        if code and code != prev_code:
            return code
        time.sleep(0.3)
    return None

def wait_until_element_enabled(element, timeout=15):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if element.tag_name == "button":
                disabled = element.get_attribute("disabled")
                aria_disabled = element.get_attribute("aria-disabled")
                if (not disabled or disabled == "false") and (not aria_disabled or aria_disabled == "false"):
                    return True
            else:
                readonly = element.get_attribute("readonly")
                disabled = element.get_attribute("disabled")
                contenteditable = element.get_attribute("contenteditable")
                if (not disabled or disabled == "false") and (not readonly or readonly == "false") and (contenteditable == "true" or element.tag_name == "textarea"):
                    return True
        except Exception:
            pass
        time.sleep(0.2)
    return False

def run_selenium_swagger():
    print("Starting Selenium automation...")
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        service = Service(CHROMEDRIVER_PATH)
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(SWAGGER_URL)
        time.sleep(3)
        section = find_swagger_endpoint(driver)
        if not section:
            print("❌ Could not find /generate/ endpoint in Swagger UI")
            return
        print("✅ Found /generate/ endpoint")

        try:
            expand_btn = section.find_element(By.CLASS_NAME, "opblock-summary")
            if "is-open" not in section.get_attribute("class"):
                expand_btn.click()
                time.sleep(1)
        except Exception:
            pass

        try:
            try_it_out_btn = section.find_element(By.XPATH, ".//button[contains(text(), 'Try it out')]")
            if "try-out__btn" in try_it_out_btn.get_attribute("class"):
                try_it_out_btn.click()
                time.sleep(1)
        except Exception:
            pass

        for i, prompt in enumerate(prompts):
            print(f"\n[UI {i+1}/10] Processing: {prompt}")

            textarea = find_request_body_input(section)
            if not textarea:
                print(f"[UI {i+1}/10] ❌ Could not find request body input for prompt '{prompt}'")
                continue
            if not wait_until_element_enabled(textarea, timeout=15):
                print(f"[UI {i+1}/10] ❌ Request input not enabled in time, skipping.")
                continue

            execute_btn = find_execute_button(section)
            if not execute_btn:
                print(f"[UI {i+1}/10] ❌ Could not find Execute button for prompt '{prompt}'")
                continue
            if not wait_until_element_enabled(execute_btn, timeout=15):
                print(f"[UI {i+1}/10] ❌ Execute button not enabled in time, skipping.")
                continue

            json_body = json.dumps({"prompt": prompt, "priority": priority}, indent=2)
            clear_and_fill_textarea(textarea, json_body)

            prev_status_code = find_status_code(section)
            execute_btn.click()
            print(f"[UI {i+1}/10] ✅ Submitted: {prompt}")
            print(f"[UI {i+1}/10] Waiting for new HTTP status code...")

            new_code = wait_for_new_status_code(section, prev_code=prev_status_code, timeout=90)
            if new_code:
                print(f"[UI {i+1}/10] New status code: {new_code}")
                try:
                    resp_elem = section.find_element(By.CSS_SELECTOR, ".responses-inner")
                    resp_text = resp_elem.text.strip()
                    print(f"[UI {i+1}/10] Response: {resp_text[:400]}...\n{'-'*50}")
                except NoSuchElementException:
                    print(f"[UI {i+1}/10] Could not find response text")
            else:
                print(f"[UI {i+1}/10] ⚠️ No new status code detected within timeout")

            time.sleep(0.5)

        print("\n✅ All 10 prompts processed! The browser window will remain open.")
        print("You can review the responses and close the window manually when done.")
        while True:
            time.sleep(10)

    except WebDriverException as e:
        print(f"❌ WebDriver error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    run_selenium_swagger()