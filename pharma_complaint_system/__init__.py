"""
AI-Powered Pharmaceutical Complaint Management System

A comprehensive system for classifying, prioritizing, and managing
pharmaceutical customer complaints using machine learning and rule-based analysis.

Example usage:
    >>> from pharma_complaint_system import (
    ...     ComplaintClassifier,
    ...     ComplaintAnalyzer,
    ...     generate_dataset,
    ...     create_dashboard,
    ... )
    >>> # Generate training data
    >>> df = generate_dataset(3000)
    >>> # Train classifier
    >>> classifier = ComplaintClassifier()
    >>> classifier.train(df)
    >>> # Analyze a complaint
    >>> analyzer = ComplaintAnalyzer()
    >>> result = analyzer.analyze(
    ...     text="Patient had allergic reaction to Amoxicillin LOT12345",
    ...     severity="Critical",
    ...     customer_type="Hospital"
    ... )
    >>> # Create dashboard
    >>> demo = create_dashboard(classifier)
    >>> demo.launch()
"""

__version__ = "1.0.0"
__author__ = "Pharma AI Team"

from .config import (
    PRODUCTS,
    CUSTOMER_TYPES,
    CATEGORIES,
    SEVERITY_LEVELS,
    SEVERITY_WEIGHTS,
    CATEGORY_TEMPLATES,
    SEVERITY_PHRASES,
    CAPA_ACTIONS,
    OWNER_MAP,
    PRODUCT_KEYWORDS,
    SENTIMENT_WORDS,
    FLAG_TERMS,
    QUEUE_COLUMNS,
    ALL_COLUMNS,
)
from .models import ComplaintClassifier
from .analysis import ComplaintAnalyzer, AnalysisResult
from .data_generator import generate_dataset
from .dashboard import create_dashboard, ComplaintDashboard

__all__ = [
    # Core classes
    "ComplaintClassifier",
    "ComplaintAnalyzer",
    "ComplaintDashboard",
    "AnalysisResult",
    # Functions
    "generate_dataset",
    "create_dashboard",
    # Configuration constants
    "PRODUCTS",
    "CUSTOMER_TYPES",
    "CATEGORIES",
    "SEVERITY_LEVELS",
    "CAPA_ACTIONS",
    "OWNER_MAP",
]
