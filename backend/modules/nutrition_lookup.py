"""
Day 6 nutrition lookup and cache builder.

Uses USDA FoodData Central when USDA_API_KEY is available, then writes a local
CSV cache that the rest of the project can use without live API calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

NUTRIENT_ALIASES = {
    "calories": ("Energy",),
    "protein": ("Protein",),
    "carbs": ("Carbohydrate, by difference",),
    "fat": ("Total lipid (fat)",),
}

QUERY_OVERRIDES = {
    "bell_pepper": "peppers, sweet, green, raw",
    "eggs": "egg, whole, raw, fresh",
    "milk": "milk, whole, 3.25% milkfat",
    "potato": "potatoes, flesh and skin, raw",
    "spinach": "spinach, raw",
    "tomato": "tomatoes, red, ripe, raw",
    "yogurt": "yogurt, plain, whole milk",
}

FIELDNAMES = [
    "class_name",
    "description",
    "calories",
    "protein",
    "carbs",
    "fat",
    "source",
    "notes",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_target_classes(class_mapping_path: Path) -> list[str]:
    with class_mapping_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    classes = data.get("target_classes")
    if not classes:
        classes = sorted(set(data.get("mapping", {}).values()))

    return list(classes)


def _extract_nutrient(food: dict[str, Any], nutrient_names: tuple[str, ...]) -> float | None:
    for nutrient in food.get("foodNutrients", []):
        name = nutrient.get("nutrientName")
        if name in nutrient_names:
            value = nutrient.get("value")
            if value is not None:
                return float(value)
    return None


def get_nutrition(food_name: str, api_key: str | None = None) -> dict[str, Any] | None:
    """Return calories, protein, carbs, and fat per 100g from USDA search."""
    api_key = api_key or os.getenv("USDA_API_KEY")
    if not api_key:
        return None

    query = QUERY_OVERRIDES.get(food_name, food_name.replace("_", " "))
    params = {"query": query, "api_key": api_key, "pageSize": 1}
    response = requests.get(USDA_SEARCH_URL, params=params, timeout=20)
    response.raise_for_status()

    foods = response.json().get("foods", [])
    if not foods:
        return None

    food = foods[0]
    return {
        "description": food.get("description"),
        "calories": _extract_nutrient(food, NUTRIENT_ALIASES["calories"]),
        "protein": _extract_nutrient(food, NUTRIENT_ALIASES["protein"]),
        "carbs": _extract_nutrient(food, NUTRIENT_ALIASES["carbs"]),
        "fat": _extract_nutrient(food, NUTRIENT_ALIASES["fat"]),
        "source": "USDA FoodData Central",
        "notes": f"query={query}",
    }


def load_manual_cache(cache_path: Path) -> dict[str, dict[str, str]]:
    if not cache_path.exists():
        return {}

    with cache_path.open("r", encoding="utf-8", newline="") as file:
        return {row["class_name"]: row for row in csv.DictReader(file)}


def build_nutrition_cache(
    class_mapping_path: Path,
    output_path: Path,
    prefer_api: bool = True,
) -> list[dict[str, Any]]:
    load_dotenv(project_root() / ".env")

    classes = load_target_classes(class_mapping_path)
    manual_cache = load_manual_cache(output_path)
    rows: list[dict[str, Any]] = []

    for class_name in classes:
        api_data = get_nutrition(class_name) if prefer_api else None
        cached_data = manual_cache.get(class_name)

        if api_data:
            rows.append({"class_name": class_name, **api_data})
        elif cached_data:
            rows.append(cached_data)
        else:
            rows.append(
                {
                    "class_name": class_name,
                    "description": "",
                    "calories": "",
                    "protein": "",
                    "carbs": "",
                    "fat": "",
                    "source": "missing",
                    "notes": "Needs manual nutrition value",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def validate_cache(class_mapping_path: Path, nutrition_path: Path) -> list[str]:
    expected_classes = set(load_target_classes(class_mapping_path))
    rows = load_manual_cache(nutrition_path)
    errors: list[str] = []

    missing_classes = sorted(expected_classes - set(rows))
    extra_classes = sorted(set(rows) - expected_classes)
    if missing_classes:
        errors.append(f"Missing nutrition rows: {', '.join(missing_classes)}")
    if extra_classes:
        errors.append(f"Unexpected nutrition rows: {', '.join(extra_classes)}")

    for class_name, row in rows.items():
        for field in ("description", "calories", "protein", "carbs", "fat"):
            if not row.get(field):
                errors.append(f"{class_name} is missing {field}")

    return errors


def main() -> int:
    root = project_root()
    parser = argparse.ArgumentParser(description="Build or validate nutrition_db.csv")
    parser.add_argument("--no-api", action="store_true", help="Use only existing manual cache values")
    parser.add_argument("--validate", action="store_true", help="Validate cache coverage after building")
    parser.add_argument(
        "--class-mapping",
        type=Path,
        default=root / "data" / "processed" / "class_mapping.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "nutrition_db.csv",
    )
    args = parser.parse_args()

    rows = build_nutrition_cache(args.class_mapping, args.output, prefer_api=not args.no_api)
    print(f"Wrote {len(rows)} nutrition rows to {args.output}")

    if args.validate:
        errors = validate_cache(args.class_mapping, args.output)
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 1
        print("Nutrition cache validation passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
