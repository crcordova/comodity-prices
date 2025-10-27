import asyncio
import json
from typing import List
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, BrowserConfig, LLMConfig, LLMExtractionStrategy
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# ----- MODELOS -----
class CommodityDescription(BaseModel):
    commodity: str = Field(..., description="Nombre del commodity")
    source_url: str = Field(..., description="URL de origen")
    description: str = Field(..., description="Descripción extraída")
    forecast: str = Field(..., description="Descripción del pronóstico")
    

# ----- EXTRACCIÓN -----
async def extract_description(url: str, commodity: str) -> CommodityDescription:
    browser_conf = BrowserConfig(headless=True)
    async with AsyncWebCrawler(config=browser_conf) as crawler:
        result = await crawler.arun(url=url)

        soup = BeautifulSoup(result.html, "html.parser")
        # # Guardar HTML para depuración
        # html_filename = f"{commodity.lower()}_fit.html"
        # with open(html_filename, "w", encoding="utf-8") as f:
        #     f.write(result.html or "")

        desc_tag = soup.select_one("h2#description")
        description = desc_tag.get_text(separator=" ", strip=True) if desc_tag else ""

        forecast_tag = soup.select_one("div#forecast-desc h3")
        forecast = forecast_tag.get_text(separator=" ", strip=True) if forecast_tag else ""


        return CommodityDescription(
            commodity=commodity,
            source_url=url,
            description=description,
            forecast=forecast
        )


# ----- MAIN -----
async def main():
    base_url = "https://tradingeconomics.com/commodity/"
    commodities = ["copper", "zinc"]

    tasks = [
        extract_description(f"{base_url}{c}", c.capitalize())
        for c in commodities
    ]

    results: List[CommodityDescription] = await asyncio.gather(*tasks)

    output_dict = {
        r.commodity: {
            "source_url": r.source_url,
            "description": r.description,
            "forecast": r.forecast
        }
        for r in results
    }

    # Guardar en JSON
    with open("commodity_descriptions.json", "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())
