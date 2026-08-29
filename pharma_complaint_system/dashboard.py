"""
Interactive dashboard for the Pharma Complaint Management System.

This module provides a Gradio-based web interface for:
- Submitting new complaints
- Viewing complaint queue
- Analytics and model performance
- System documentation
"""

import datetime
import html as html_lib
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt

try:
    import gradio as gr
except ImportError:
    raise ImportError(
        "Gradio is required for the dashboard. Install it with: pip install gradio"
    )

from .config import (
    PRODUCTS,
    CUSTOMER_TYPES,
    CATEGORIES,
    SEVERITY_LEVELS,
    CAPA_ACTIONS,
    OWNER_MAP,
    QUEUE_COLUMNS,
    ALL_COLUMNS,
)
from .models import ComplaintClassifier
from .analysis import ComplaintAnalyzer


class ComplaintDashboard:
    """
    Gradio-based dashboard for pharmaceutical complaint management.

    This class creates an interactive web interface with tabs for:
    - New complaint submission with AI analysis
    - Complaint queue viewing and filtering
    - Analytics charts and model metrics
    - System documentation

    Attributes:
        classifier: Trained ComplaintClassifier instance
        analyzer: ComplaintAnalyzer instance
        complaints_df: DataFrame storing all submitted complaints

    Example:
        >>> from pharma_complaint_system import ComplaintClassifier, generate_dataset
        >>> df = generate_dataset(3000)
        >>> classifier = ComplaintClassifier()
        >>> classifier.train(df)
        >>> dashboard = ComplaintDashboard(classifier)
        >>> demo = dashboard.create()
        >>> demo.launch()
    """

    def __init__(self, classifier: ComplaintClassifier):
        """
        Initialize the dashboard with a trained classifier.

        Args:
            classifier: A trained ComplaintClassifier instance.

        Raises:
            ValueError: If the classifier is not trained.
        """
        if classifier.category_model is None or classifier.severity_model is None:
            raise ValueError("Classifier must be trained before creating dashboard.")

        self.classifier = classifier
        self.analyzer = ComplaintAnalyzer()
        self.complaints_df = pd.DataFrame(columns=ALL_COLUMNS)
        self._complaint_counter = 0

    def _generate_complaint_id(self) -> str:
        """Generate a unique complaint ID."""
        self._complaint_counter += 1
        year = datetime.datetime.now().year
        return f"CMP-{year}-{self._complaint_counter:04d}"

    def _get_queue_df(
        self,
        severity_filter: str = "All",
        category_filter: str = "All",
    ) -> pd.DataFrame:
        """Get filtered complaint queue DataFrame."""
        if self.complaints_df.empty:
            return pd.DataFrame(columns=QUEUE_COLUMNS)

        filtered_df = self.complaints_df.copy()

        if severity_filter != "All":
            filtered_df = filtered_df[filtered_df["Severity"] == severity_filter]

        if category_filter != "All":
            filtered_df = filtered_df[filtered_df["Category"] == category_filter]

        if filtered_df.empty:
            return pd.DataFrame(columns=QUEUE_COLUMNS)

        filtered_df = filtered_df.sort_values("Date", ascending=False)
        return filtered_df[QUEUE_COLUMNS].reset_index(drop=True)

    def _render_analysis_html(self, analysis: dict[str, Any]) -> str:
        """Render analysis results as HTML."""
        e = html_lib.escape

        top_categories = "".join([
            f"<li>{e(name)}: {prob:.1%}</li>"
            for name, prob in analysis["top_categories"]
        ])

        top_severities = "".join([
            f"<li>{e(name)}: {prob:.1%}</li>"
            for name, prob in analysis["top_severities"]
        ])

        flags_html = "".join([
            f"<li>{e(flag)}</li>"
            for flag in analysis["flags"]
        ]) or "<li>No major regulatory flags detected.</li>"

        capa_html = "".join([
            f"<li>{e(action)}</li>"
            for action in analysis["capa"]
        ])

        severity_color = {
            "Critical": "#ffcdd2",
            "Major": "#ffe0b2",
            "Minor": "#c8e6c9",
        }.get(analysis["severity"], "#e0e0e0")

        return f"""
        <div style="border:1px solid #d0d7de; border-radius:12px; padding:18px; background:#ffffff; font-family: Arial, sans-serif;">
            <h2 style="margin-top:0;">AI Complaint Assessment</h2>

            <p>
                <strong>Complaint ID:</strong> {e(analysis['id'])}<br>
                <strong>Date:</strong> {e(analysis['date'])}<br>
                <strong>Product:</strong> {e(analysis['product'])}<br>
                <strong>Batch/Lot:</strong> {e(analysis['batch'])}<br>
                <strong>Customer Type:</strong> {e(analysis['customer_type'])}<br>
                <strong>Reporter:</strong> {e(analysis['reporter'])}
            </p>

            <p>
                <strong>AI Category:</strong>
                <span style="background:#e3f2fd; padding:4px 10px; border-radius:8px; font-weight:bold;">
                    {e(analysis['category'])}
                </span>
            </p>

            <p>
                <strong>AI Severity:</strong>
                <span style="background:{severity_color}; padding:4px 10px; border-radius:8px; font-weight:bold;">
                    {e(analysis['severity'])}
                </span>
            </p>

            <p>
                <strong>Sentiment:</strong> {e(analysis['sentiment'])}<br>
                <strong>Priority:</strong> {e(analysis['priority_level'])} ({analysis['priority_score']}/100)<br>
                <strong>Suggested Owner:</strong> {e(analysis['owner'])}
            </p>

            <h3>Top Category Predictions</h3>
            <ul>{top_categories}</ul>

            <h3>Top Severity Predictions</h3>
            <ul>{top_severities}</ul>

            <h3>Regulatory / Quality Flags</h3>
            <ul>{flags_html}</ul>

            <h3>Suggested CAPA Actions</h3>
            <ul>{capa_html}</ul>
        </div>
        """

    def _make_charts(self) -> tuple[plt.Figure, plt.Figure]:
        """Create analytics charts."""
        if self.complaints_df.empty:
            return (
                self._create_empty_figure("No complaints submitted yet"),
                self._create_empty_figure("No complaints submitted yet"),
            )

        # Category chart
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        category_counts = self.complaints_df["Category"].value_counts()
        category_counts.plot(kind="bar", ax=ax1, color="steelblue")
        ax1.set_title("Complaints by Category")
        ax1.set_xlabel("Category")
        ax1.set_ylabel("Count")
        plt.setp(ax1.get_xticklabels(), rotation=25, ha="right")
        fig1.tight_layout()

        # Severity chart
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        severity_order = [
            sev for sev in ["Critical", "Major", "Minor"]
            if sev in self.complaints_df["Severity"].values
        ]
        severity_counts = self.complaints_df["Severity"].value_counts().reindex(severity_order)
        severity_counts.plot(
            kind="bar",
            ax=ax2,
            colors=["#d9534f", "#f0ad4e", "#5cb85c"][:len(severity_order)]
        )
        ax2.set_title("Complaints by Severity")
        ax2.set_xlabel("Severity")
        ax2.set_ylabel("Count")
        fig2.tight_layout()

        return fig1, fig2

    def _create_empty_figure(self, message: str) -> plt.Figure:
        """Create an empty figure with a message."""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    def _analyze_and_submit(
        self,
        text: str,
        product: str,
        batch: str,
        customer_type: str,
        reporter: str,
        severity_filter: str,
        category_filter: str,
    ) -> tuple[str, str, pd.DataFrame, plt.Figure, plt.Figure]:
        """Process and submit a new complaint."""
        if not text or not text.strip():
            fig1, fig2 = self._make_charts()
            return (
                "⚠️ Please enter a complaint description.",
                "<p style='color:red;'>No complaint text provided.</p>",
                self._get_queue_df(severity_filter, category_filter),
                fig1,
                fig2,
            )

        # Get AI predictions
        top_categories = self.classifier._top_predictions(
            self.classifier.category_model, text, n=3
        )
        top_severities = self.classifier._top_predictions(
            self.classifier.severity_model, text, n=3
        )

        predicted_category = top_categories[0][0]
        predicted_severity = top_severities[0][0]

        # Extract/resolve fields
        final_product = product
        if final_product == "Auto-detect":
            final_product = self.analyzer.detect_product(text) or "Unspecified"

        final_batch = batch.strip() if batch and batch.strip() else (
            self.analyzer.extract_batch(text) or "Not provided"
        )

        # Analyze sentiment and priority
        sentiment, _ = self.analyzer.analyze_sentiment(text)
        flags = self.analyzer.detect_regulatory_flags(text)
        priority = self.analyzer.calculate_priority(
            predicted_severity, sentiment, customer_type, flags
        )

        owner = self.analyzer.get_suggested_owner(predicted_category)
        capa = self.analyzer.get_capa_actions(predicted_category)

        # Create complaint record
        complaint_id = self._generate_complaint_id()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        row = {
            "ID": complaint_id,
            "Date": now,
            "Description": text.strip(),
            "Product": final_product,
            "Batch": final_batch,
            "Customer Type": customer_type,
            "Reporter": reporter or "Not provided",
            "Category": predicted_category,
            "Severity": predicted_severity,
            "Sentiment": sentiment,
            "Priority Score": priority[0],
            "Priority Level": priority[1],
            "Status": "Open",
            "Owner": owner,
        }

        self.complaints_df = pd.concat(
            [self.complaints_df, pd.DataFrame([row])],
            ignore_index=True,
        )

        # Prepare analysis display
        analysis = {
            "id": complaint_id,
            "date": now,
            "product": final_product,
            "batch": final_batch,
            "customer_type": customer_type,
            "reporter": reporter or "Not provided",
            "category": predicted_category,
            "severity": predicted_severity,
            "sentiment": sentiment,
            "priority_score": priority[0],
            "priority_level": priority[1],
            "owner": owner,
            "flags": flags,
            "capa": capa,
            "top_categories": top_categories,
            "top_severities": top_severities,
        }

        status_message = (
            f"✅ **{complaint_id}** saved.  \n"
            f"AI Triage: **{predicted_category}** | Severity: **{predicted_severity}** | "
            f"Priority: **{priority[1]}** ({priority[0]}/100)"
        )

        fig1, fig2 = self._make_charts()

        return (
            status_message,
            self._render_analysis_html(analysis),
            self._get_queue_df(severity_filter, category_filter),
            fig1,
            fig2,
        )

    def create(self) -> gr.Blocks:
        """
        Create the Gradio dashboard interface.

        Returns:
            A configured Gradio Blocks application.
        """
        explanation_md = """
## How this AI system works

### 1. Text input
The user enters a customer complaint in natural language.

### 2. AI classification
Two machine learning models predict:
- Complaint category
- Complaint severity

### 3. Rule-based enhancement
The system also applies business rules to:
- detect product name
- extract batch/lot number
- detect sentiment
- generate regulatory flags
- calculate priority score
- suggest responsible department
- suggest CAPA actions

### 4. Dashboard
The complaint is stored in an in-memory complaint queue and shown in the dashboard.

### Note
This demo uses synthetic training data and in-memory storage.
For production, use a secure database, authentication, audit logs, validated models, and pharma compliance controls.
"""

        initial_cat_plot, initial_sev_plot = self._make_charts()
        metrics_df = self.classifier.get_metrics_dataframe()

        with gr.Blocks(title="AI Pharma Complaint Management System") as demo:
            gr.Markdown("""
            # AI-Powered Customer Complaint Management System
            ### Pharmaceutical Manufacturing Intern Project Demo
            """)

            with gr.Tabs():
                # Tab 1: New Complaint
                with gr.Tab("📝 New Complaint"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            complaint_text = gr.Textbox(
                                label="Complaint Description",
                                lines=6,
                                placeholder=(
                                    "Example: Hospital received multiple blister packs of "
                                    "Amoxicillin 250mg Capsule LOT56789 with broken seals "
                                    "and missing leaflets. This is a serious packaging issue "
                                    "and may affect patient safety."
                                ),
                            )

                            product_input = gr.Dropdown(
                                ["Auto-detect"] + PRODUCTS,
                                label="Product",
                                value="Auto-detect",
                            )

                            batch_input = gr.Textbox(
                                label="Batch/Lot Number",
                                placeholder="LOT12345",
                            )

                            customer_type_input = gr.Dropdown(
                                CUSTOMER_TYPES,
                                label="Customer/Reporter Type",
                                value="Patient",
                            )

                            reporter_input = gr.Textbox(
                                label="Reporter Name / ID (optional)",
                                placeholder="Enter reporter name or ticket ID",
                            )

                            submit_btn = gr.Button(
                                "AI Analyze & Submit", variant="primary"
                            )

                            gr.Markdown("""
                            **Try this example:**

                            Hospital received multiple blister packs of Amoxicillin 250mg Capsule LOT56789 with broken seals and missing leaflets. This is a serious packaging issue and may affect patient safety.
                            """)

                        with gr.Column(scale=3):
                            status_msg = gr.Markdown("")
                            analysis_html = gr.HTML("")

                    submit_btn.click(
                        fn=self._analyze_and_submit,
                        inputs=[
                            complaint_text,
                            product_input,
                            batch_input,
                            customer_type_input,
                            reporter_input,
                            gr.State("All"),
                            gr.State("All"),
                        ],
                        outputs=[
                            status_msg,
                            analysis_html,
                            gr.Dataframe(),
                            gr.Plot(),
                            gr.Plot(),
                        ],
                    )

                # Tab 2: Complaint Queue
                with gr.Tab("📋 Complaint Queue"):
                    with gr.Row():
                        filter_severity = gr.Dropdown(
                            ["All"] + SEVERITY_LEVELS,
                            label="Filter by Severity",
                            value="All",
                        )

                        filter_category = gr.Dropdown(
                            ["All"] + CATEGORIES,
                            label="Filter by Category",
                            value="All",
                        )

                        refresh_btn = gr.Button("Refresh Queue")

                    queue_table = gr.Dataframe(
                        value=self._get_queue_df("All", "All"),
                        interactive=False,
                        label="Complaint Queue",
                    )

                    refresh_btn.click(
                        fn=self._get_queue_df,
                        inputs=[filter_severity, filter_category],
                        outputs=queue_table,
                    )

                # Tab 3: Analytics
                with gr.Tab("📊 Analytics"):
                    analytics_refresh = gr.Button("Refresh Analytics")

                    with gr.Row():
                        cat_plot = gr.Plot(
                            label="Complaints by Category",
                            value=initial_cat_plot,
                        )
                        sev_plot = gr.Plot(
                            label="Complaints by Severity",
                            value=initial_sev_plot,
                        )

                    metrics_table = gr.Dataframe(
                        value=metrics_df,
                        label="Model Performance",
                        interactive=False,
                    )

                    cat_acc = self.classifier.category_accuracy
                    sev_acc = self.classifier.severity_accuracy
                    gr.Markdown(
                        f"**Category accuracy:** {cat_acc:.2%} | **Severity accuracy:** {sev_acc:.2%}"
                    )

                    analytics_refresh.click(
                        fn=self._make_charts,
                        inputs=None,
                        outputs=[cat_plot, sev_plot],
                    )

                # Tab 4: Documentation
                with gr.Tab("ℹ️ How It Works"):
                    gr.Markdown(explanation_md)

        return demo


def create_dashboard(classifier: ComplaintClassifier) -> gr.Blocks:
    """
    Convenience function to create a dashboard.

    Args:
        classifier: A trained ComplaintClassifier instance.

    Returns:
        A configured Gradio Blocks application.

    Example:
        >>> classifier = ComplaintClassifier()
        >>> classifier.train(generate_dataset(3000))
        >>> demo = create_dashboard(classifier)
        >>> demo.launch()
    """
    dashboard = ComplaintDashboard(classifier)
    return dashboard.create()
