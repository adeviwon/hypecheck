"""Headless regression checks for the current input demo and repaired archived pages.

Optional QA dependencies: pip install playwright; playwright install chromium firefox webkit
Run: python qa_playwright.py --base-url http://127.0.0.1:8000
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

from playwright.async_api import async_playwright


async def main(base_url, output):
    output.mkdir(parents=True, exist_ok=True)
    results = {"base_url": base_url, "runs": [], "layout_repeats": [], "limitations": ["Headless viewport emulation, not physical-device certification.", "Local synthetic performance, not field Core Web Vitals."]}
    async with async_playwright() as playwright:
        for engine in ["chromium", "firefox", "webkit"]:
            browser = await getattr(playwright, engine).launch()
            for device, width, height in [("desktop", 1920, 1080), ("tablet", 768, 1024), ("mobile", 375, 667)]:
                page = await browser.new_page(viewport={"width": width, "height": height})
                row = {"engine": engine, "device": device, "errors": []}
                results["runs"].append(row)
                page.on("pageerror", lambda error, row=row: row["errors"].append(str(error)))
                start = time.perf_counter()
                await page.goto(base_url + "/predict.html")
                await page.wait_for_function('() => !document.querySelector("#forecast").disabled')
                row["ready_ms"] = round(1000 * (time.perf_counter() - start), 2)
                assert await page.locator("#coin option").count() == 13
                row["screenshots"] = [f"predict-{engine}-{device}-before.png"]
                await page.screenshot(path=str(output / row["screenshots"][-1]), full_page=True)
                await page.locator("#forecast").click()
                assert "8–4,000" in await page.locator("#result").inner_text()
                await page.select_option("#coin", "DOGE")
                await page.fill("#tweet", "DOGE is breaking resistance with increasing trading volume.")
                await page.locator("#forecast").click()
                text = await page.locator("#result").inner_text()
                assert "47.7%" in text and "DOWN / FLAT" in text
                row["tweet_only_result"] = text
                row["screenshots"].append(f"predict-{engine}-{device}-after.png")
                await page.screenshot(path=str(output / row["screenshots"][-1]), full_page=True)
                await page.fill("#context", "Earlier public discussion about the coin and price.")
                await page.locator("#forecast").click()
                row["context_result"] = await page.locator("#result").inner_text()
                assert "Estimated chance" in row["context_result"]
                assert row["context_result"] != row["tweet_only_result"]
                await page.fill("#tweet", "<script>alert('untrusted source text')</script>")
                await page.locator("#forecast").click()
                assert await page.locator("#result script").count() == 0
                assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                assert not row["errors"], row["errors"]
                row["status"] = "PASS"
                await page.close()
            await browser.close()
        browser = await playwright.chromium.launch()
        for path, width, height in [("headlines.html", 768, 1024), ("index.html", 1920, 1080)]:
            for iteration in range(3):
                page = await browser.new_page(viewport={"width": width, "height": height})
                await page.add_init_script('window.qaCLS=0;new PerformanceObserver(list=>{for(const entry of list.getEntries())if(!entry.hadRecentInput)qaCLS+=entry.value}).observe({type:"layout-shift",buffered:true})')
                await page.goto(base_url + "/" + path)
                await page.wait_for_function('() => document.querySelector("#coin").options.length>0')
                await page.wait_for_timeout(400)
                value = await page.evaluate("qaCLS")
                results["layout_repeats"].append({"page": path, "iteration": iteration + 1, "CLS": value})
                assert value <= .1, (path, value)
                if path == "index.html":
                    await page.locator('a[href="predict.html"]').click()
                    await page.wait_for_function('() => !document.querySelector("#forecast").disabled')
                    assert page.url.endswith("/predict.html")
                await page.close()
        page = await browser.new_page(viewport={"width": 375, "height": 667})
        await page.goto(base_url + "/headlines.html")
        await page.wait_for_function('() => document.querySelector("#coin").options.length>0')
        region = page.locator('.scroll[aria-label^="Post category"]')
        await region.focus()
        await page.keyboard.press("ArrowRight")
        await page.wait_for_timeout(150)
        results["keyboard_table_scroll_px"] = await region.evaluate("element=>element.scrollLeft")
        assert results["keyboard_table_scroll_px"] > 0
        await region.screenshot(path=str(output / "mobile-table-keyboard-scrolled.png"))
        await page.close()
        page = await browser.new_page()
        await page.route("**/tweet-forecast-model.json", lambda route: route.fulfill(status=503, body="unavailable"))
        await page.goto(base_url + "/predict.html")
        await page.wait_for_function('() => document.querySelector("#forecast").textContent==="Model unavailable"')
        assert await page.locator("#forecast").is_disabled()
        results["model_failure"] = await page.locator("#result").inner_text()
        await page.screenshot(path=str(output / "predict-model-failure.png"), full_page=True)
        await browser.close()
    results["screenshots"] = [{"file": file.name, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()} for file in sorted(output.glob("*.png"))]
    results["status"] = "PASS"
    (output / "predict-test-results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("public/qa-screenshots"))
    args = parser.parse_args()
    asyncio.run(main(args.base_url.rstrip("/"), args.output))
