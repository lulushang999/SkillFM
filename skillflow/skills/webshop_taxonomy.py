"""Deterministic WebShop task taxonomy shared by data and evaluation paths."""

from __future__ import annotations

import re


WEBSHOP_TASK_TYPE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "apparel",
        (
            "shirt",
            "t shirt",
            "tee",
            "blouse",
            "dress",
            "skirt",
            "pants",
            "trousers",
            "jeans",
            "shorts",
            "jacket",
            "coat",
            "sweater",
            "hoodie",
            "sweatshirt",
            "clothing",
        ),
    ),
    (
        "footwear",
        (
            "shoe",
            "sneaker",
            "boot",
            "sandal",
            "slipper",
            "heel",
            "loafer",
            "cleat",
            "footwear",
        ),
    ),
    (
        "electronics",
        (
            "electronics",
            "laptop",
            "computer",
            "keyboard",
            "mouse",
            "monitor",
            "headphone",
            "earbud",
            "speaker",
            "camera",
            "phone",
            "smartphone",
            "tablet",
            "charger",
            "adapter",
            "television",
            "smartwatch",
        ),
    ),
    (
        "accessories",
        (
            "accessory",
            "accessories",
            "bag",
            "backpack",
            "purse",
            "handbag",
            "wallet",
            "belt",
            "hat",
            "cap",
            "scarf",
            "scarves",
            "sunglasses",
            "watch",
            "jewelry",
            "jewellery",
            "necklace",
            "bracelet",
            "earring",
            "ring",
        ),
    ),
    (
        "home_decor",
        (
            "home decor",
            "furniture",
            "chair",
            "table",
            "desk",
            "sofa",
            "couch",
            "bedding",
            "bedsheet",
            "pillow",
            "blanket",
            "rug",
            "curtain",
            "lamp",
            "vase",
            "mirror",
            "wall art",
            "poster",
            "picture frame",
        ),
    ),
    (
        "beauty_health",
        (
            "beauty",
            "health",
            "makeup",
            "cosmetic",
            "skincare",
            "skin care",
            "lotion",
            "moisturizer",
            "shampoo",
            "conditioner",
            "soap",
            "perfume",
            "fragrance",
            "deodorant",
            "toothbrush",
            "toothpaste",
            "supplement",
            "vitamin",
        ),
    ),
)


def classify_webshop_task_type(goal: str) -> str:
    """Return the shared, deterministic WebShop routing category."""

    normalized = re.sub(r"[^a-z0-9]+", " ", str(goal).casefold()).strip()
    padded = f" {normalized} "
    words = set(normalized.split())
    for task_type, keywords in WEBSHOP_TASK_TYPE_KEYWORDS:
        for keyword in keywords:
            if " " in keyword and f" {keyword} " in padded:
                return task_type
            if keyword in words or f"{keyword}s" in words or f"{keyword}es" in words:
                return task_type
    return "other"


__all__ = ["WEBSHOP_TASK_TYPE_KEYWORDS", "classify_webshop_task_type"]
