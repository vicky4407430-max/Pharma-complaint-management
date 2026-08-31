"""
Complaint analysis with rule-based enhancements.

This module provides the ComplaintAnalyzer class that performs:
- Product detection from text
- Batch/lot number extraction
- Sentiment analysis
- Regulatory flag detection
- Priority score calculation
- CAPA action suggestions
"""

import re
from dataclasses import dataclass, field

from .config import (
    PRODUCT_KEYWORDS,
    SENTIMENT_WORDS,
    FLAG_TERMS,
    CAPA_ACTIONS,
    OWNER_MAP,
    SEVERITY_LEVELS,
)


@dataclass
class AnalysisResult:
    """Container for complaint analysis results."""
    product: str | None
    batch: str | None
    sentiment: str
    sentiment_score: int
    flags: list[str]
    priority_score: int
    priority_level: str
    owner: str
    capa_actions: list[str]


class ComplaintAnalyzer:
    """
    Rule-based analyzer for pharmaceutical complaints.

    This class provides methods to extract structured information from
    complaint text using pattern matching and business rules.

    Example:
        >>> analyzer = ComplaintAnalyzer()
        >>> result = analyzer.analyze(
        ...     text="Patient had allergic reaction to Amoxicillin LOT12345",
        ...     severity="Critical",
        ...     customer_type="Hospital"
        ... )
        >>> print(result.priority_level)
        'Immediate'
    """

    def __init__(self):
        """Initialize the analyzer with compiled regex patterns."""
        self._batch_patterns = [
            re.compile(r"batch\s*[:#-]?\s*([A-Za-z0-9]{4,20})", re.IGNORECASE),
            re.compile(r"lot\s*[:#-]?\s*([A-Za-z0-9]{4,20})", re.IGNORECASE),
            re.compile(r"\b(LOT\d{4,10})\b"),
            re.compile(r"\b(B\d{4,10})\b"),
        ]

    def detect_product(self, text: str) -> str | None:
        """
        Detect product name from complaint text.

        Args:
            text: The complaint text to analyze.

        Returns:
            The detected product name, or None if no product is detected.
        """
        if not text:
            return None

        t = text.lower()

        for keyword, product in PRODUCT_KEYWORDS.items():
            if keyword in t:
                return product

        return None

    def extract_batch(self, text: str) -> str | None:
        """
        Extract batch/lot number from complaint text.

        Args:
            text: The complaint text to analyze.

        Returns:
            The extracted batch number, or None if no batch is found.
        """
        if not text:
            return None

        for pattern in self._batch_patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).upper()

        return None

    def analyze_sentiment(self, text: str) -> tuple[str, int]:
        """
        Analyze sentiment of complaint text.

        Args:
            text: The complaint text to analyze.

        Returns:
            Tuple of (sentiment_label, raw_score) where:
                - sentiment_label is "Positive", "Neutral", or "Negative"
                - raw_score is neg_count - pos_count
        """
        t = text.lower()

        neg_count = sum(1 for word in SENTIMENT_WORDS["negative"] if word in t)
        pos_count = sum(1 for word in SENTIMENT_WORDS["positive"] if word in t)

        score = neg_count - pos_count

        if score >= 2:
            label = "Negative"
        elif score <= -1:
            label = "Positive"
        else:
            label = "Neutral"

        return label, score

    def detect_regulatory_flags(self, text: str) -> list[str]:
        """
        Detect regulatory and quality flags in complaint text.

        Args:
            text: The complaint text to analyze.

        Returns:
            List of detected flag labels.
        """
        t = text.lower()
        flags = []

        for label, terms in FLAG_TERMS.items():
            if any(term in t for term in terms):
                flags.append(label)

        return flags

    def calculate_priority(
        self,
        severity: str,
        sentiment: str,
        customer_type: str,
        flags: list[str],
    ) -> tuple[int, str]:
        """
        Calculate priority score and level for a complaint.

        Args:
            severity: Complaint severity (Critical, Major, Minor).
            sentiment: Sentiment label (Positive, Neutral, Negative).
            customer_type: Type of customer/reporter.
            flags: List of regulatory flags detected.

        Returns:
            Tuple of (priority_score, priority_level) where:
                - priority_score is 0-100
                - priority_level is "Immediate", "High", "Medium", or "Low"
        """
        # Base score from severity
        base_scores = {
            "Critical": 80,
            "Major": 55,
            "Minor": 25,
        }
        base_score = base_scores.get(severity, 35)

        # Sentiment adjustment
        sentiment_scores = {
            "Negative": 10,
            "Neutral": 5,
            "Positive": 0,
        }
        sentiment_score = sentiment_scores.get(sentiment, 0)

        # Customer type adjustment
        customer_scores = {
            "Hospital": 12,
            "Physician": 12,
            "Patient": 10,
            "Pharmacist": 8,
            "Distributor": 6,
        }
        customer_score = customer_scores.get(customer_type, 4)

        # Flag adjustment (capped at 15)
        flag_score = min(len(flags) * 5, 15)

        # Calculate total (capped at 100)
        total_score = min(base_score + sentiment_score + customer_score + flag_score, 100)

        # Determine priority level
        if total_score >= 85:
            level = "Immediate"
        elif total_score >= 70:
            level = "High"
        elif total_score >= 45:
            level = "Medium"
        else:
            level = "Low"

        return total_score, level

    def get_suggested_owner(self, category: str) -> str:
        """
        Get the suggested department owner for a complaint category.

        Args:
            category: The complaint category.

        Returns:
            The suggested owner department.
        """
        return OWNER_MAP.get(category, "Quality Assurance")

    def get_capa_actions(self, category: str) -> list[str]:
        """
        Get suggested CAPA (Corrective and Preventive Action) for a category.

        Args:
            category: The complaint category.

        Returns:
            List of suggested CAPA actions.
        """
        default_action = ["Initiate quality investigation."]
        return CAPA_ACTIONS.get(category, default_action)

    def analyze(
        self,
        text: str,
        severity: str,
        customer_type: str,
        category: str | None = None,
    ) -> AnalysisResult:
        """
        Perform comprehensive analysis of a complaint.

        Args:
            text: The complaint text to analyze.
            severity: The complaint severity level.
            customer_type: The type of customer/reporter.
            category: Optional complaint category for CAPA suggestions.

        Returns:
            AnalysisResult dataclass with all analysis findings.
        """
        product = self.detect_product(text)
        batch = self.extract_batch(text)
        sentiment, sentiment_score = self.analyze_sentiment(text)
        flags = self.detect_regulatory_flags(text)
        priority_score, priority_level = self.calculate_priority(
            severity, sentiment, customer_type, flags
        )

        owner = self.get_suggested_owner(category) if category else "Quality Assurance"
        capa_actions = self.get_capa_actions(category) if category else ["Initiate quality investigation."]

        return AnalysisResult(
            product=product,
            batch=batch,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            flags=flags,
            priority_score=priority_score,
            priority_level=priority_level,
            owner=owner,
            capa_actions=capa_actions,
        )
