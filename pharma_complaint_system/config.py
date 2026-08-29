"""
Configuration constants for the Pharma Complaint Management System.

This module contains all configuration data including:
- Product and customer type definitions
- Complaint categories and severity levels
- Text templates for data generation
- CAPA actions and ownership mappings
- Sentiment analysis keywords
- Regulatory flag terms
"""

# =============================================================================
# CORE ENTITIES
# =============================================================================

PRODUCTS = [
    "Paracetamol 500mg Tablet",
    "Amoxicillin 250mg Capsule",
    "Metformin 500mg Tablet",
    "Insulin Glargine Injection",
    "Omeprazole 20mg Capsule",
    "Salbutamol Inhaler",
    "Cetirizine Syrup",
    "Amlodipine 5mg Tablet",
]

CUSTOMER_TYPES = [
    "Patient",
    "Pharmacist",
    "Hospital",
    "Physician",
    "Distributor",
]

CATEGORIES = [
    "Product Quality",
    "Packaging Defect",
    "Labeling Issue",
    "Contamination",
    "Stability/Potency",
    "Adverse Event",
    "Supply/Logistics",
    "Efficacy Concern",
]

SEVERITY_LEVELS = ["Critical", "Major", "Minor"]

# =============================================================================
# DATA GENERATION CONFIGURATION
# =============================================================================

SEVERITY_WEIGHTS: dict[str, list[float]] = {
    "Product Quality": [0.10, 0.60, 0.30],
    "Packaging Defect": [0.15, 0.55, 0.30],
    "Labeling Issue": [0.25, 0.55, 0.20],
    "Contamination": [0.75, 0.20, 0.05],
    "Stability/Potency": [0.45, 0.45, 0.10],
    "Adverse Event": [0.75, 0.20, 0.05],
    "Supply/Logistics": [0.15, 0.55, 0.30],
    "Efficacy Concern": [0.25, 0.60, 0.15],
}

CATEGORY_TEMPLATES: dict[str, list[str]] = {
    "Product Quality": [
        "Customer received {product} from batch {batch} with chipped, cracked, or broken tablets.",
        "Tablets in {product} batch {batch} appear discolored and have unusual spots.",
        "Capsules in {product} batch {batch} are sticking together and some are empty.",
        "Patient found damaged dosage units in {product} batch {batch}.",
        "Pharmacist reported crumbling tablets while dispensing {product} batch {batch}.",
    ],
    "Packaging Defect": [
        "Blister pack of {product} batch {batch} has broken seal and exposed tablets.",
        "Carton of {product} batch {batch} was crushed and primary pack is damaged.",
        "Missing patient information leaflet inside pack of {product} batch {batch}.",
        "Child-resistant cap on {product} batch {batch} is not closing properly.",
        "Leaking bottle observed for {product} batch {batch} during pharmacy receipt.",
    ],
    "Labeling Issue": [
        "Expiry date printed on {product} batch {batch} label is smudged and unreadable.",
        "Label claim on {product} batch {batch} does not match the product inside.",
        "Wrong strength mentioned on carton of {product} batch {batch}.",
        "Batch number and expiry mismatch on label of {product} batch {batch}.",
        "Missing warning sticker on {product} batch {batch} pack.",
    ],
    "Contamination": [
        "Foreign black particles found in {product} batch {batch}.",
        "Visible mold or microbial growth suspected in {product} batch {batch}.",
        "Unusual chemical odor noticed when opening {product} batch {batch}.",
        "Glass fragments reported near container of {product} batch {batch}.",
        "Hair or fiber contamination observed in {product} batch {batch} pack.",
    ],
    "Stability/Potency": [
        "{product} batch {batch} was exposed to high temperature during storage.",
        "Pharmacist suspects potency loss because patients report poor response to {product} batch {batch}.",
        "Cold chain temperature excursion occurred for {product} batch {batch}.",
        "Product {product} batch {batch} changed color after storage.",
        "Dissolution concern raised for {product} batch {batch} after routine checks.",
    ],
    "Adverse Event": [
        "Patient experienced severe rash and breathing difficulty after taking {product} batch {batch}.",
        "Hospitalization reported due to suspected reaction to {product} batch {batch}.",
        "Patient reports unexpected bleeding after using {product} batch {batch}.",
        "Overdose risk because patient misunderstood dosing of {product} batch {batch}.",
        "Serious allergic reaction suspected with {product} batch {batch}.",
    ],
    "Supply/Logistics": [
        "Shipment of {product} batch {batch} arrived later than required and temperature log missing.",
        "Wrong quantity delivered for {product} batch {batch}.",
        "Pallet of {product} batch {batch} was dropped during warehouse handling.",
        "Distributor received damaged outer cases of {product} batch {batch}.",
        "Cold chain shipment for {product} batch {batch} showed temperature excursion.",
    ],
    "Efficacy Concern": [
        "Patient states {product} batch {batch} is no longer controlling blood glucose.",
        "Physician reports lack of therapeutic response with {product} batch {batch}.",
        "Patient says blood pressure remains high despite taking {product} batch {batch}.",
        "Pharmacist received multiple complaints that {product} batch {batch} seems ineffective.",
        "Patient switched from {product} batch {batch} and symptoms improved, suggesting product issue.",
    ],
}

SEVERITY_PHRASES: dict[str, list[str]] = {
    "Critical": [
        "This is a serious patient safety issue requiring immediate escalation.",
        "Potential recall and regulatory notification may be needed.",
        "Risk of severe harm, hospitalization, or death.",
        "Urgent quality event with high regulatory impact.",
    ],
    "Major": [
        "This is a significant GMP deviation requiring formal investigation.",
        "Potential nonconformance and CAPA required.",
        "Multiple units or patients may be affected.",
        "Requires quality review before further distribution.",
    ],
    "Minor": [
        "This appears to be a minor cosmetic issue with low patient risk.",
        "Single isolated occurrence with limited impact.",
        "No immediate patient safety concern identified.",
        "Can be handled through routine quality review.",
    ],
}

# =============================================================================
# BUSINESS RULES CONFIGURATION
# =============================================================================

CAPA_ACTIONS: dict[str, list[str]] = {
    "Adverse Event": [
        "Notify Pharmacovigilance and Safety team within 24 hours.",
        "Collect patient outcome, concomitant medications, and medical history.",
        "Assess need for expedited regulatory reporting.",
        "Quarantine retained samples and review batch records.",
    ],
    "Contamination": [
        "Immediately quarantine affected batch and distribution stock.",
        "Initiate microbiological or foreign matter testing.",
        "Review environmental monitoring, line clearance, and filtration records.",
        "Assess batch recall risk and notify Quality Assurance.",
    ],
    "Product Quality": [
        "Retain samples and perform visual/physical testing.",
        "Review manufacturing, compression/coating, and in-process controls.",
        "Inspect transport and storage conditions.",
        "Determine if OOS investigation or CAPA is required.",
    ],
    "Packaging Defect": [
        "Check packaging line records, seal integrity, and cartoning operations.",
        "Perform AQL inspection of retained samples.",
        "Review transport damage and secondary packaging.",
        "Correct labeling/artwork if involved.",
    ],
    "Labeling Issue": [
        "Verify label version, expiry, batch overprinting, and artwork approval.",
        "Check labeling line reconciliation and line clearance.",
        "Assess risk of medication error and notify Regulatory Affairs.",
        "Issue market notification if labeling is noncompliant.",
    ],
    "Stability/Potency": [
        "Initiate OOS/stability investigation.",
        "Test retained samples for assay, dissolution, and degradation products.",
        "Review storage conditions and cold chain data.",
        "Assess shelf-life and possible recall.",
    ],
    "Supply/Logistics": [
        "Review distribution records, temperature logs, and chain of custody.",
        "Investigate carrier handling and cold chain excursion.",
        "Assess product impact and replace/credit if needed.",
        "Implement preventive controls with logistics partner.",
    ],
    "Efficacy Concern": [
        "Verify authenticity, storage, and expiry.",
        "Review batch release and stability data.",
        "Collect clinical details and concomitant therapy.",
        "Evaluate possible subpotency or drug interaction.",
    ],
}

OWNER_MAP: dict[str, str] = {
    "Adverse Event": "Pharmacovigilance",
    "Contamination": "Quality Assurance / Microbiology",
    "Product Quality": "Quality Assurance / Production",
    "Packaging Defect": "Packaging QA",
    "Labeling Issue": "Regulatory Affairs / QA",
    "Stability/Potency": "Quality Control / Stability",
    "Supply/Logistics": "Supply Chain QA",
    "Efficacy Concern": "Medical Affairs / QA",
}

PRODUCT_KEYWORDS: dict[str, str] = {
    "paracetamol": "Paracetamol 500mg Tablet",
    "acetaminophen": "Paracetamol 500mg Tablet",
    "amoxicillin": "Amoxicillin 250mg Capsule",
    "metformin": "Metformin 500mg Tablet",
    "insulin": "Insulin Glargine Injection",
    "glargine": "Insulin Glargine Injection",
    "omeprazole": "Omeprazole 20mg Capsule",
    "salbutamol": "Salbutamol Inhaler",
    "albuterol": "Salbutamol Inhaler",
    "cetirizine": "Cetirizine Syrup",
    "amlodipine": "Amlodipine 5mg Tablet",
}

SENTIMENT_WORDS: dict[str, list[str]] = {
    "negative": [
        "angry", "upset", "disappointed", "danger", "unsafe", "severe",
        "serious", "urgent", "hospital", "injury", "death", "allergic",
        "contamination", "wrong", "damaged", "broken", "defective",
        "expired", "leak", "ineffective", "failure", "concern", "risk", "adverse",
    ],
    "positive": [
        "good", "satisfied", "excellent", "thank", "no issue", "resolved",
    ],
}

FLAG_TERMS: dict[str, list[str]] = {
    "Adverse event": ["adverse", "allergic", "reaction", "hospital", "injury", "death", "overdose"],
    "Contamination": ["contamination", "foreign", "particle", "mold", "microbial", "glass", "hair"],
    "Labeling risk": ["wrong label", "wrong strength", "wrong product", "expiry", "unreadable", "missing leaflet"],
    "Recall potential": ["recall", "market action", "serious patient safety"],
    "Cold chain": ["cold chain", "temperature excursion", "temperature log"],
}

# =============================================================================
# DATAFRAME SCHEMA
# =============================================================================

QUEUE_COLUMNS: list[str] = [
    "ID",
    "Date",
    "Product",
    "Batch",
    "Customer Type",
    "Category",
    "Severity",
    "Priority Level",
    "Priority Score",
    "Status",
    "Owner",
]

ALL_COLUMNS: list[str] = [
    "ID",
    "Date",
    "Description",
    "Product",
    "Batch",
    "Customer Type",
    "Reporter",
    "Category",
    "Severity",
    "Sentiment",
    "Priority Score",
    "Priority Level",
    "Status",
    "Owner",
]
