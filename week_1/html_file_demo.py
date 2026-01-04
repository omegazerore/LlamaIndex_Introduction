import asyncio
from playwright.async_api import async_playwright

url = "https://warhammer40k.fandom.com/wiki/Titan"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")

        html = await page.content()

        with open("Titan.html", "w", encoding="utf-8") as f:
            f.write(html)

        await browser.close()

    print("Saved Titan.html")


if __name__ == "__main__":

    asyncio.run(main())