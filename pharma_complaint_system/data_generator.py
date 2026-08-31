"""
Synthetic data generation for training the complaint classification models.

This module provides functionality to generate realistic pharmaceutical
complaint data for training and testing machine learning models.
"""

import random
from typing import Literal

import pandas as pd

from .config import (
    CATEGORIES,
    PRODUCTS,
    SEVERITY_LEVELS,
    SEVERITY_WEIGHTS,
    CATEGORY_TEMPLATES,
    SEVERITY_PHRASES,
)


def generate_dataset(
    n: int = 3000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic dataset of pharmaceutical complaints.

    Args:
        n: Number of samples to generate. Default is 3000.
        seed: Random seed for reproducibility. Default is 42.

    Returns:
        A pandas DataFrame with columns:
            - text: The complaint description
            - category: The complaint category
            - severity: The severity level (Critical, Major, Minor)

    Example:
        >>> df = generate_dataset(100)
        >>> df.head()
    """
    random.seed(seed)

    rows: list[dict[str, str]] = []

    for _ in range(n):
        # Select category and product
        category = random.choice(CATEGORIES)
        product = random.choice(PRODUCTS)
        batch = f"LOT{random.randint(10000, 99999)}"

        # Generate complaint text from template
        template = random.choice(CATEGORY_TEMPLATES[category])
        text = template.format(product=product, batch=batch)

        # Determine severity based on category weights
        weights = SEVERITY_WEIGHTS[category]
        severity = random.choices(SEVERITY_LEVELS, weights=weights, k=1)[0]

        # Add severity phrase to text
        text += " " + random.choice(SEVERITY_PHRASES[severity])

        rows.append({
            "text": text,
            "category": category,
            "severity": severity,
        })

    return pd.DataFrame(rows)
