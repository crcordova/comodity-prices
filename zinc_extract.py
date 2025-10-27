


import asyncio
from playwright.async_api import async_playwright
import pandas as pd
from functions import upload_s3

URL = "https://markets.businessinsider.com/commodities/zinc-price?op=1"

async def get_zinc_data():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)  # headless=False para ver qué hace
        context = await browser.new_context()
        page = await context.new_page()

        data_holder = {"data": None}

        # Capturar todas las respuestas y depurar
        async def handle_response(response):
            url = response.url
            if "Chart_GetChartData" in url:
                try:
                    text = await response.text()
                    print(f"   → Tamaño respuesta: {len(text)} chars")
                    # Intentar parsear JSON
                    data = await response.json()
                    if data and len(data) > 300:  # heurística: 5 años trae muchos registros
                        data_holder["data"] = data
                        print(f"✅ Posible histórico largo capturado ({len(data)} filas)")
                except Exception as e:
                    print("⚠️ No se pudo parsear como JSON:", e)

        page.on("response", handle_response)

        # Navegar a la página
        await page.goto(URL, wait_until="domcontentloaded")

        # Hacer click en el tab "5y"
        await page.locator("text=5y").click()

        # Esperar un poco para que lleguen los requests
        await page.wait_for_timeout(8000)

        if data_holder["data"]:
            df = pd.DataFrame(data_holder["data"])
            print(f"Total registros guardados: {len(df)}")
            df.to_csv("historical_data/Zinc_prices.csv", index=False)
        else:
            print("❌ No se capturó data de 5 años, revisa las URLs impresas")

        await browser.close()
        upload_s3(df, "prices/Zinc_prices.csv")

# Ejecutar
asyncio.run(get_zinc_data())
