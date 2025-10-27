import asyncio
import json
from typing import List
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, BrowserConfig

# ----- MODELOS -----
class ForecastPoint(BaseModel):
    date: str = Field(..., description="Fecha del forecast (ej: Q3/25)")
    value: str = Field(..., description="Valor pronosticado")

class CommodityForecast(BaseModel):
    commodity: str
    source_url: str
    forecast: List[ForecastPoint]

# ----- EXTRACCIÓN -----
async def extract_forecast(url: str, commodity: str) -> CommodityForecast:
    browser_conf = BrowserConfig(headless=True)
    async with AsyncWebCrawler(config=browser_conf) as crawler:
        result = await crawler.arun(url=url)

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(result.fit_html, "html.parser")
        rows = soup.select("table tr")

        header = soup.select_one("table tr")
        ths = header.find_all("th")
        quarters = [th.get_text(strip=True) for th in ths if th.get_text(strip=True)]
        quarters = quarters[-4:]  # nos quedamos con los últimos 4

        # con fines de depuración
        # with open("debug_page.html", "w", encoding="utf-8") as f:
        #     f.write(result.fit_html)

        forecast_points: List[ForecastPoint] = []
        for row in rows:
            cols = row.find_all("td")
            if not cols:
                continue
            name_tag = row.select_one("td.datatable-item-first b")
            if not name_tag:
                continue

            name = name_tag.get_text(strip=True)
            if name.lower() != commodity.lower():
                continue
            values = [c.get_text(strip=True).replace(",", "") for c in cols[3:]]
            for i, val in enumerate(values[:len(quarters)]):
                if val:
                    forecast_points.append(ForecastPoint(date=quarters[i], value=val))

        return CommodityForecast(commodity=commodity, source_url=url, forecast=forecast_points)

# ----- MAIN -----
async def main():
    url = "https://tradingeconomics.com/forecast/commodity"
    commodities = ["Copper", "Zinc"]

    results: Dict[str, Any] = {}
    for c in commodities:
        forecast = await extract_forecast(url, c)
        results[c] = {
            "source_url": forecast.source_url,
            "forecast": [fp.model_dump() for fp in forecast.forecast],
        }

    # Guardar en JSON
    with open("commodity_forecasts.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())
