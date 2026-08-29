"""
Machine learning models for complaint classification.

This module provides the ComplaintClassifier class that handles:
- Training category and severity classification models
- Predicting categories and severities from complaint text
- Model evaluation and reporting
"""

from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import SEVERITY_LEVELS


PredictionResult: TypeAlias = list[tuple[str, float]]


class ComplaintClassifier:
    """
    A classifier for pharmaceutical complaints that predicts both
    category and severity from complaint text.

    Attributes:
        category_model: Pipeline for predicting complaint category
        severity_model: Pipeline for predicting complaint severity
        category_accuracy: Accuracy score on test set for category model
        severity_accuracy: Accuracy score on test set for severity model

    Example:
        >>> from pharma_complaint_system import ComplaintClassifier, generate_dataset
        >>> df = generate_dataset(3000)
        >>> classifier = ComplaintClassifier()
        >>> classifier.train(df)
        >>> predictions = classifier.predict_category("Tablets are broken")
    """

    def __init__(
        self,
        max_features: int = 20000,
        ngram_range: tuple[int, int] = (1, 2),
        max_iter: int = 1000,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initialize the classifier with configurable parameters.

        Args:
            max_features: Maximum number of TF-IDF features. Default is 20000.
            ngram_range: Range of n-grams for TF-IDF. Default is (1, 2).
            max_iter: Maximum iterations for logistic regression. Default is 1000.
            test_size: Proportion of data to use for testing. Default is 0.2.
            random_state: Random seed for reproducibility. Default is 42.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_iter = max_iter
        self.test_size = test_size
        self.random_state = random_state

        self.category_model: Pipeline | None = None
        self.severity_model: Pipeline | None = None
        self.category_accuracy: float | None = None
        self.severity_accuracy: float | None = None

        # Store training data for potential re-evaluation
        self._X_test: pd.Series | None = None
        self._y_cat_test: pd.Series | None = None
        self._y_sev_test: pd.Series | None = None

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Train both category and severity classification models.

        Args:
            df: DataFrame with 'text', 'category', and 'severity' columns.

        Returns:
            Dictionary containing:
                - category_accuracy: Accuracy of category classifier
                - severity_accuracy: Accuracy of severity classifier

        Raises:
            ValueError: If required columns are missing from DataFrame.
        """
        required_columns = {"text", "category", "severity"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        X = df["text"]
        y_cat = df["category"]
        y_sev = df["severity"]

        # Split data with stratification on category
        X_train, X_test, y_cat_train, y_cat_test, y_sev_train, y_sev_test = (
            train_test_split(
                X,
                y_cat,
                y_sev,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_cat,
            )
        )

        # Store test data for later evaluation
        self._X_test = X_test
        self._y_cat_test = y_cat_test
        self._y_sev_test = y_sev_test

        # Create and train category model
        self.category_model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                stop_words="english",
            )),
            ("clf", LogisticRegression(max_iter=self.max_iter)),
        ])

        # Create and train severity model
        self.severity_model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                stop_words="english",
            )),
            ("clf", LogisticRegression(max_iter=self.max_iter)),
        ])

        self.category_model.fit(X_train, y_cat_train)
        self.severity_model.fit(X_train, y_sev_train)

        # Evaluate models
        cat_pred = self.category_model.predict(X_test)
        sev_pred = self.severity_model.predict(X_test)

        self.category_accuracy = accuracy_score(y_cat_test, cat_pred)
        self.severity_accuracy = accuracy_score(y_sev_test, sev_pred)

        return {
            "category_accuracy": self.category_accuracy,
            "severity_accuracy": self.severity_accuracy,
        }

    def predict_category(
        self,
        text: str,
        n: int = 1,
    ) -> str | PredictionResult:
        """
        Predict the category of a complaint.

        Args:
            text: The complaint text to classify.
            n: Number of top predictions to return. If 1, returns just the category string.

        Returns:
            If n=1: The predicted category as a string.
            If n>1: List of (category, probability) tuples.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self.category_model is None:
            raise RuntimeError("Model must be trained before making predictions.")

        if n == 1:
            return self.category_model.predict([text])[0]
        else:
            return self._top_predictions(self.category_model, text, n)

    def predict_severity(
        self,
        text: str,
        n: int = 1,
    ) -> str | PredictionResult:
        """
        Predict the severity of a complaint.

        Args:
            text: The complaint text to classify.
            n: Number of top predictions to return. If 1, returns just the severity string.

        Returns:
            If n=1: The predicted severity as a string.
            If n>1: List of (severity, probability) tuples.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self.severity_model is None:
            raise RuntimeError("Model must be trained before making predictions.")

        if n == 1:
            return self.severity_model.predict([text])[0]
        else:
            return self._top_predictions(self.severity_model, text, n)

    def _top_predictions(
        self,
        model: Pipeline,
        text: str,
        n: int = 3,
    ) -> PredictionResult:
        """Get top N predictions with probabilities."""
        probabilities = model.predict_proba([text])[0]
        classes = model.classes_

        top_items = sorted(
            zip(classes, probabilities),
            key=lambda x: x[1],
            reverse=True,
        )[:n]

        return [(str(cls), float(prob)) for cls, prob in top_items]

    def evaluate(self, verbose: bool = True) -> dict:
        """
        Evaluate model performance on test data.

        Args:
            verbose: If True, print detailed classification reports.

        Returns:
            Dictionary containing accuracy scores and classification reports.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self.category_model is None or self._X_test is None:
            raise RuntimeError("Model must be trained before evaluation.")

        cat_pred = self.category_model.predict(self._X_test)
        sev_pred = self.severity_model.predict(self._X_test)

        results = {
            "category_accuracy": accuracy_score(self._y_cat_test, cat_pred),
            "severity_accuracy": accuracy_score(self._y_sev_test, sev_pred),
            "category_report": classification_report(
                self._y_cat_test, cat_pred, output_dict=True,
            ),
            "severity_report": classification_report(
                self._y_sev_test, sev_pred, output_dict=True,
            ),
        }

        if verbose:
            print(f"Category Accuracy: {results['category_accuracy']:.4f}")
            print(f"\nCategory Classification Report:")
            print(classification_report(self._y_cat_test, cat_pred))

            print(f"\nSeverity Accuracy: {results['severity_accuracy']:.4f}")
            print(f"\nSeverity Classification Report:")
            print(classification_report(self._y_sev_test, sev_pred))

        return results

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Get model metrics as a pandas DataFrame.

        Returns:
            DataFrame with model names, accuracies, and purposes.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self.category_accuracy is None or self.severity_accuracy is None:
            raise RuntimeError("Model must be trained before getting metrics.")

        return pd.DataFrame({
            "Model": ["Category Classifier", "Severity Classifier"],
            "Test Accuracy": [
                round(self.category_accuracy, 4),
                round(self.severity_accuracy, 4),
            ],
            "Purpose": [
                "Predict complaint type from text",
                "Predict complaint severity from text",
            ],
        })
