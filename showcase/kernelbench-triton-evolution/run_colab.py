#!/usr/bin/env python3
"""
Automate running the KernelBench Evolution notebook on Google Colab.

Uses browser_cookie3 to extract Google session from Chrome (like xfeed does for X).

Usage:
    python run_colab.py [--headless] [--timeout MINUTES]
"""

import argparse
import asyncio
import json
import re
import random
import time
from datetime import datetime
from pathlib import Path

import browser_cookie3
from playwright.async_api import async_playwright, Page


# Paths
NOTEBOOK_PATH = Path(__file__).parent / "KernelBench_Evolution.ipynb"
RESULTS_DIR = Path(__file__).parent / "results"


# Timing settings (like xfeed)
PAGE_LOAD_MIN = 1000
PAGE_LOAD_MAX = 2000
ACTION_DELAY_MIN = 500
ACTION_DELAY_MAX = 1000


def _page_load_delay() -> int:
    return random.randint(PAGE_LOAD_MIN, PAGE_LOAD_MAX)


def _action_delay() -> int:
    return random.randint(ACTION_DELAY_MIN, ACTION_DELAY_MAX)


def get_google_session_from_chrome() -> list[dict]:
    """Get Google session cookies from Chrome browser."""
    try:
        cookies = []

        # Get google.com cookies
        cj = browser_cookie3.chrome(domain_name=".google.com")
        for cookie in cj:
            cookies.append({
                "name": cookie.name,
                "value": cookie.value,
                "domain": cookie.domain,
                "path": cookie.path,
                "secure": bool(cookie.secure),
                "httpOnly": bool(cookie.has_nonstandard_attr("HttpOnly")),
            })

        # Also get colab.research.google.com specific cookies
        try:
            cj_colab = browser_cookie3.chrome(domain_name="colab.research.google.com")
            for cookie in cj_colab:
                cookies.append({
                    "name": cookie.name,
                    "value": cookie.value,
                    "domain": cookie.domain,
                    "path": cookie.path,
                    "secure": bool(cookie.secure),
                    "httpOnly": bool(cookie.has_nonstandard_attr("HttpOnly")),
                })
        except Exception:
            pass  # May not have Colab-specific cookies yet

        if not cookies:
            raise RuntimeError("No Google cookies found")

        return cookies
    except Exception as e:
        raise RuntimeError(
            f"Could not access Chrome session: {e}\n"
            "Make sure you are logged into Google in Chrome."
        )


async def wait_for_colab_ready(page: Page, timeout: int = 60000):
    """Wait for Colab to fully load."""
    print("  Waiting for Colab to load...")

    # Wait for page to settle
    try:
        await page.wait_for_load_state('networkidle', timeout=30000)
    except Exception:
        pass

    await page.wait_for_timeout(_page_load_delay())

    # Try various Colab indicators
    selectors = [
        'colab-toolbar',
        '#toolbar',
        'text=File',
        'text=Runtime',
        'text=Welcome to Colaboratory',
        'text=Open notebook',
    ]

    for selector in selectors:
        try:
            await page.wait_for_selector(selector, timeout=10000)
            print(f"  Found: {selector}")
            await page.wait_for_timeout(_action_delay())
            return
        except Exception:
            continue

    # Take screenshot for debugging
    screenshot_path = RESULTS_DIR / "debug_screenshot.png"
    screenshot_path.parent.mkdir(exist_ok=True)
    await page.screenshot(path=str(screenshot_path))
    print(f"  Screenshot saved to: {screenshot_path}")

    # Don't fail - try to continue anyway
    print("  Warning: Could not detect Colab interface, continuing...")


async def connect_to_gpu(page: Page, timeout: int = 120000):
    """Connect to a GPU runtime."""
    print("  Connecting to GPU runtime...")

    # Click on "Connect" button or runtime menu
    try:
        connect_btn = page.locator('colab-connect-button')
        if await connect_btn.is_visible():
            await connect_btn.click()
            await page.wait_for_timeout(_action_delay())
    except Exception:
        pass

    # Open Runtime menu
    await page.click('text=Runtime')
    await page.wait_for_timeout(_action_delay())

    # Click "Change runtime type"
    await page.click('text=Change runtime type')
    await page.wait_for_timeout(1000)

    # Select GPU from dropdown
    try:
        await page.wait_for_selector('text=Hardware accelerator', timeout=5000)

        # Try selecting T4 GPU
        try:
            await page.click('text=T4 GPU')
        except Exception:
            # Try dropdown approach
            dropdown = page.locator('select').filter(has_text='None').first
            if await dropdown.is_visible():
                await dropdown.select_option('gpu')
    except Exception:
        print("  Warning: Could not select GPU, may already be selected")

    await page.wait_for_timeout(_action_delay())

    # Click Save
    try:
        await page.click('text=Save')
    except Exception:
        try:
            await page.click('button:has-text("Save")')
        except Exception:
            pass

    await page.wait_for_timeout(2000)

    # Wait for runtime to connect
    print("  Waiting for GPU runtime to connect (this may take 30-60s)...")

    try:
        await page.wait_for_selector('text=T4', timeout=timeout)
        print("  GPU runtime connected!")
    except Exception:
        # Check if already connected
        if await page.locator('text=RAM').is_visible():
            print("  Runtime connected (checking GPU...)")
        else:
            print("  Warning: Could not confirm GPU connection")


async def run_all_cells(page: Page, timeout: int = 600000):
    """Run all cells in the notebook."""
    print("  Running all cells...")

    # Use keyboard shortcut Ctrl+F9 (Run all) on Mac it's Cmd+F9
    await page.keyboard.press('Meta+F9')  # Mac

    await page.wait_for_timeout(2000)

    # If there's a confirmation dialog, accept it
    try:
        await page.click('text=Run anyway', timeout=3000)
    except Exception:
        pass

    print("  Cells are running... (this may take 5-10 minutes)")

    # Wait for execution to complete
    start_time = time.time()
    max_wait = timeout / 1000

    while time.time() - start_time < max_wait:
        # Check for the busy indicator
        try:
            busy = await page.locator('[aria-label="Busy"]').is_visible()
            if not busy:
                await page.wait_for_timeout(5000)
                busy = await page.locator('[aria-label="Busy"]').is_visible()
                if not busy:
                    break
        except Exception:
            pass

        elapsed = int(time.time() - start_time)
        print(f"    Still running... ({elapsed}s elapsed)")
        await page.wait_for_timeout(10000)

    print("  All cells completed!")


async def extract_results(page: Page) -> tuple[dict, str]:
    """Extract results from the notebook output."""
    print("  Extracting results...")

    results = {
        "gpu": None,
        "baseline": {},
        "generations": {},
        "best_generation": None,
        "best_speedup": None,
        "timestamp": datetime.now().isoformat(),
    }

    # Get all cell outputs
    outputs = await page.locator('.output_text, .output_stdout').all_text_contents()
    full_output = "\n".join(outputs)

    # Parse GPU info
    gpu_match = re.search(r'GPU: (.+)', full_output)
    if gpu_match:
        results["gpu"] = gpu_match.group(1)

    # Parse baseline timings
    baseline_matches = re.findall(r'(\d+x\d+)[^:]*:\s*([\d.]+)ms', full_output)
    for shape, time_ms in baseline_matches:
        if shape not in results["baseline"]:
            results["baseline"][shape] = float(time_ms)

    # Parse speedups for each generation
    gen_pattern = r'GENERATION (\d+)[^\n]*\n.*?Speedup vs PyTorch:\s*([\d.]+)x'
    gen_matches = re.findall(gen_pattern, full_output, re.DOTALL)
    for gen, speedup in gen_matches:
        if f"Gen {gen}" not in results["generations"]:
            results["generations"][f"Gen {gen}"] = {"speedup": float(speedup)}

    # Parse best generation
    best_match = re.search(r'Best Generation:\s*(.+)', full_output)
    if best_match:
        results["best_generation"] = best_match.group(1).strip()

    speedup_match = re.search(r'Average Speedup:\s*([\d.]+)x', full_output)
    if speedup_match:
        results["best_speedup"] = float(speedup_match.group(1))

    return results, full_output


def save_results(results: dict, full_output: str):
    """Save results to files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    results_path = RESULTS_DIR / "colab_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {results_path}")

    # Save full output
    output_path = RESULTS_DIR / "colab_output.txt"
    with open(output_path, 'w') as f:
        f.write(full_output)
    print(f"  Full output saved to: {output_path}")


async def run_notebook_on_colab(headless: bool = False, timeout_minutes: int = 15):
    """Main function to run the notebook on Colab."""
    print("\n" + "=" * 60)
    print("KernelBench Triton Evolution - Colab Automation")
    print("=" * 60)

    # Get session from Chrome
    print("\n[1/6] Getting Google session from Chrome...")
    cookies = get_google_session_from_chrome()
    print(f"  Found {len(cookies)} cookies")

    results = {}

    async with async_playwright() as p:
        print("\n[2/6] Launching browser...")

        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        # Add cookies to the context
        await context.add_cookies(cookies)

        page = await context.new_page()

        # Navigate to Colab
        print("\n[3/6] Opening Google Colab...")
        await page.goto("https://colab.research.google.com/")

        try:
            await wait_for_colab_ready(page)
        except Exception as e:
            print(f"\n  Colab issue: {e}")
            print("   Continuing anyway...")

        # Check if we're logged in
        if "/signin" in page.url or "accounts.google.com" in page.url:
            print("\n  ERROR: Not logged in to Google.")
            print("  Please log in to Google in Chrome first, then try again.")
            await browser.close()
            return {}

        # Upload notebook
        print("\n[4/6] Uploading notebook...")

        # Colab shows "Open notebook" dialog by default - use it directly
        # First check if the dialog is open
        open_dialog = page.locator('mwc-dialog.colab-open-dialog')
        if await open_dialog.is_visible():
            print("  Found Open notebook dialog, using it...")

            # Click "Upload" tab
            try:
                await page.click('text=Upload', timeout=5000)
                await page.wait_for_timeout(_action_delay())
            except Exception:
                pass

            # Handle file upload - look for upload area
            try:
                async with page.expect_file_chooser() as fc_info:
                    # Click on the upload area or browse button
                    upload_area = page.locator('text=browse').first
                    if await upload_area.is_visible():
                        await upload_area.click()
                    else:
                        # Try clicking the upload input directly
                        await page.click('input[type="file"]', force=True)
                file_chooser = fc_info.value
                await file_chooser.set_files(str(NOTEBOOK_PATH))
            except Exception as e:
                print(f"  Upload via dialog failed: {e}")
                # Close dialog and try menu approach
                await page.press('body', 'Escape')
                await page.wait_for_timeout(1000)

        # If no dialog or upload failed, try File menu approach
        if not await page.locator('text=KernelBench').is_visible():
            print("  Trying File menu approach...")
            await page.click('text=File')
            await page.wait_for_timeout(_action_delay())

            await page.click('text=Upload notebook')
            await page.wait_for_timeout(1000)

            async with page.expect_file_chooser() as fc_info:
                await page.click('text=Browse')
            file_chooser = fc_info.value
            await file_chooser.set_files(str(NOTEBOOK_PATH))

        await page.wait_for_timeout(3000)
        await wait_for_colab_ready(page)

        # Connect to GPU
        print("\n[5/6] Setting up GPU runtime...")
        await connect_to_gpu(page)

        # Run all cells
        print("\n[6/6] Executing notebook...")
        await run_all_cells(page, timeout=timeout_minutes * 60 * 1000)

        # Extract results
        print("\n[7/7] Collecting results...")
        results, full_output = await extract_results(page)
        save_results(results, full_output)

        # Take final screenshot
        screenshot_path = RESULTS_DIR / "final_screenshot.png"
        await page.screenshot(path=str(screenshot_path))
        print(f"  Final screenshot: {screenshot_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"GPU: {results.get('gpu', 'Unknown')}")
        print(f"Best Generation: {results.get('best_generation', 'Unknown')}")
        print(f"Best Speedup: {results.get('best_speedup', 'Unknown')}x")
        print("=" * 60)

        # Keep browser open briefly for inspection if not headless
        if not headless:
            print("\nBrowser will close in 10 seconds...")
            await page.wait_for_timeout(10000)

        await browser.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Run KernelBench Evolution on Colab")
    parser.add_argument('--headless', action='store_true',
                        help='Run browser in headless mode')
    parser.add_argument('--timeout', type=int, default=15,
                        help='Timeout in minutes for notebook execution (default: 15)')

    args = parser.parse_args()

    results = asyncio.run(run_notebook_on_colab(
        headless=args.headless,
        timeout_minutes=args.timeout
    ))

    return 0 if results.get('best_speedup') else 1


if __name__ == "__main__":
    exit(main())
