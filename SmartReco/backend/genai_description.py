"""
genai_description.py — Offline AI-style product description generator.
"""
import random

def enhance_description(name: str, category: str = "", material: str = "", brand: str = "") -> str:
    name = name.strip().title() if name else "This product"
    category = category.lower() if category else "home item"
    material = material.lower() if material else "premium materials"
    brand = brand.strip().title() if brand else "our trusted brand"

    templates = [
        f"The {name} by {brand} is a beautifully crafted {category} made from high-quality {material}, offering long-lasting durability and modern design.",
        f"Experience premium craftsmanship with {brand}'s {name}. This {category} features fine {material} construction and a timeless look.",
        f"{name} combines elegant design with sturdy {material} build — a perfect choice for your {category} needs.",
        f"Bring elegance to your space with the {name} by {brand}, crafted using durable {material} and designed for everyday comfort.",
        f"{brand} presents the {name}, a perfect blend of design and functionality. Built with top-grade {material}, ideal for modern {category} settings.",
    ]
    return random.choice(templates)
