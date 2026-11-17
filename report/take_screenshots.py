"""
Captures the web-app screenshots used in the report (report/screenshots/).

    python report/take_screenshots.py

Starts the FastAPI server and the React dev server locally, then drives
Microsoft Edge (already installed on Windows) with Playwright.
"""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = Path(__file__).resolve().parent / "screenshots"
API_PORT, WEB_PORT = 8011, 5181
API = f"http://127.0.0.1:{API_PORT}"
WEB = f"http://127.0.0.1:{WEB_PORT}"


def wait_for(url, seconds=90):
    end = time.time() + seconds
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=3)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"{url} did not start")


def start_servers():
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--port", str(API_PORT)],
                           cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    env = dict(os.environ, VITE_API_URL=API)
    npx = "npx.cmd" if os.name == "nt" else "npx"
    web = subprocess.Popen([npx, "vite", "--port", str(WEB_PORT), "--strictPort", "--host", "127.0.0.1"],
                           cwd=ROOT / "frontend", env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_for(API + "/api/health")
    wait_for(WEB)
    return api, web


def settle(page, ms=1200):
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(ms)


def pick(page, title):
    page.locator(f".thumb[title='{title}']").click()
    page.wait_for_function("() => !document.querySelector('.hero.is-loading')")
    settle(page)


def main():
    SHOTS.mkdir(exist_ok=True)
    api, web = start_servers()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page(viewport={"width": 1366, "height": 860}, device_scale_factor=1.5,
                                    color_scheme="light", reduced_motion="reduce")
            page.goto(WEB)
            page.wait_for_selector(".verdict h2", timeout=60000)
            settle(page)

            # 1. Landing view: infected cell, verdict, gallery
            page.screenshot(path=SHOTS / "01_live_diagnosis.png")
            # 2. The 10-stage grid
            page.locator(".steps").screenshot(path=SHOTS / "02_pipeline_steps.png")
            # 3. Histogram + Otsu details
            page.locator(".details").screenshot(path=SHOTS / "03_histogram_otsu.png")
            # 4. Healthy cell
            pick(page, "Uninfected #1")
            page.locator(".hero").screenshot(path=SHOTS / "04_uninfected_verdict.png")
            # 5. Tuning the pipeline (Gaussian 7x7) on an infected cell
            pick(page, "Parasitized #3")
            page.get_by_role("radio", name="Gaussian").click()
            page.get_by_role("radio", name="7").click()
            page.wait_for_function("() => !document.querySelector('.hero.is-loading')")
            settle(page)
            page.screenshot(path=SHOTS / "05_tuning_gaussian7.png")
            page.get_by_role("radio", name="Median").click()
            page.get_by_role("radio", name="3").first.click()
            # 6. Uploading an image
            page.set_input_files("input[type=file]", str(ROOT / "samples" / "parasitized_4.png"))
            page.wait_for_function("() => document.querySelector('.mini b') && "
                                   "[...document.querySelectorAll('.mini b')].some(b => b.textContent === 'Uploaded')")
            settle(page)
            page.locator(".hero").screenshot(path=SHOTS / "06_upload_result.png")
            # 7. Model performance
            page.get_by_role("button", name="Model performance").click()
            page.wait_for_selector(".metric-value")
            settle(page, 1500)
            page.evaluate("window.scrollTo(0, 0)")
            page.locator(".metric-grid").screenshot(path=SHOTS / "07_metrics.png")
            page.locator(".perf-row").screenshot(path=SHOTS / "08_confusion_curves.png")
            # 8. Syllabus map
            page.get_by_role("button", name="Syllabus map").click()
            settle(page)
            page.locator(".syllabus").screenshot(path=SHOTS / "09_syllabus_map.png")
            # 9. Phone layout
            phone = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2,
                                     color_scheme="light", is_mobile=True, has_touch=True)
            phone.goto(WEB)
            phone.wait_for_selector(".verdict h2", timeout=60000)
            settle(phone)
            phone.evaluate("document.querySelector('.hero').scrollIntoView({block: 'start'}); window.scrollBy(0, -70)")
            phone.wait_for_timeout(500)
            phone.screenshot(path=SHOTS / "10_mobile.png")
            browser.close()
    finally:
        for proc in (api, web):
            proc.terminate()
        if os.name == "nt":  # npx starts a child node process on Windows
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(web.pid)], capture_output=True)
    for f in sorted(SHOTS.glob("*.png")):
        print(f.name)


if __name__ == "__main__":
    main()
