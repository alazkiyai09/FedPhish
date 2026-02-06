# Phishing Email Analysis - Feature Engineering Pipeline

A comprehensive feature extraction pipeline for phishing email detection, specialized for **financial services fraud detection**. This project provides discriminative features for training machine learning models to identify phishing emails targeting banking customers.

## 🔑 Key Differentiator: Financial-Specific Features

Unlike generic phishing detectors, this pipeline includes **financial-specific feature extraction** for banking phishing:

- **Bank Name Impersonation Detection**: Uses Levenshtein distance to detect typosquatting of major banks (Chase, Wells Fargo, ANZ, BNZ, etc.)
- **Wire Transfer Urgency Detection**: Identifies time-sensitive wire transfer scams
- **Credential Harvesting Detection**: Detects "verify your account" patterns
- **Account/Routing Number Requests**: Flags sensitive information requests
- **Payment Invoice Terminology**: Identifies fraudulent invoice schemes

## Features

### 7 Feature Categories (60+ Features)

| Category | Features | Purpose |
|----------|----------|---------|
| **URL** | 10 features | IP-based URLs, suspicious TLDs, URL length, subdomain count |
| **Header** | 10 features | SPF/DKIM/DMARC validation, hop count, reply-to mismatch |
| **Sender** | 10 features | Freemail detection, domain reputation, display name tricks |
| **Content** | 10 features | Urgency keywords, CTA density, threat language |
| **Structural** | 10 features | HTML/text ratio, attachments, embedded images, forms |
| **Linguistic** | 10 features | Spelling errors, grammar score, formality level |
| **Financial** | 10 features | ⭐ Bank impersonation, wire urgency, credential harvesting |

## Installation

```bash
# Clone repository
cd phishing_email_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Install dev dependencies (optional)
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from src.transformers import PhishingFeaturePipeline

# Load your email data
emails_df = pd.read_csv("data/processed/emails.csv")

# Create pipeline with all default extractors
pipeline = PhishingFeaturePipeline()

# Fit and transform
features = pipeline.fit_transform(emails_df)

# View features
print(f"Extracted {features.shape[1]} features from {features.shape[0]} emails")
print(f"Feature names: {pipeline.get_feature_names()}")

# Print extraction statistics
pipeline.print_extraction_summary()
```

## Input Data Format

The pipeline expects a DataFrame with the following columns:

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `body` | str | Plain text email body | ✅ Yes |
| `headers` | dict or str | Email headers | ✅ Yes |
| `subject` | str | Email subject line | ✅ Yes |
| `from_addr` | str | Sender email address | ✅ Yes |
| `body_html` | str | HTML email body | Optional |
| `attachments` | list | Attachment metadata | Optional |

```python
# Example email data
emails_df = pd.DataFrame([
    {
        "body": "Verify your account immediately: http://chase-secure.xyz/login",
        "headers": {"Received": "...", "SPF": "pass"},
        "subject": "Urgent: Account Verification",
        "from_addr": "support@chase-secure.xyz",
        "body_html": "<p>Verify your account</p>",
        "attachments": []
    }
])
```

## Feature Documentation

### URL Features (`URLFeatureExtractor`)

- `url_count`: Number of URLs (normalized)
- `has_ip_url`: Contains IP address like `http://192.168.1.1`
- `avg_url_length`: Average URL length
- `has_suspicious_tld`: Uses `.xyz`, `.top`, `.tk` etc.
- `has_https`: Contains HTTPS URL
- `avg_subdomain_count`: Average subdomains per URL
- `has_url_shortener`: Uses bit.ly, tinyurl etc.
- `special_char_ratio`: Special characters for obfuscation
- `has_port_specified`: URL contains port number
- `max_url_length`: Maximum URL length

### Header Features (`HeaderFeatureExtractor`)

- `spf_pass`: SPF validation passed
- `spf_fail`: SPF validation failed
- `dkim_present`: DKIM signature present
- `dkim_valid`: DKIM signature valid
- `dmarc_pass`: DMARC validation passed
- `dmarc_fail`: DMARC validation failed
- `hop_count`: Number of mail servers in path
- `reply_to_mismatch`: Reply-To differs from From
- `has_priority_flag`: Marked high priority/urgent
- `has_authentication_results`: Authentication headers present

### Sender Features (`SenderFeatureExtractor`)

- `is_freemail`: Uses gmail.com, yahoo.com etc.
- `display_name_mismatch`: Display name doesn't match email
- `display_name_has_bank`: Display name contains bank name
- `domain_age_days`: Domain registration age
- `has_numbers_in_domain`: Domain contains numbers (suspicious)
- `email_address_length`: Email address length
- `domain_length`: Domain length
- `sender_name_length`: Display name length
- `has_reply_to_path`: Has Reply-To header
- `suspicious_pattern`: Matches suspicious patterns like `support@gmail.com`

### Content Features (`ContentFeatureExtractor`)

- `urgency_keyword_count`: Words like "urgent", "immediately"
- `cta_button_count`: "Click here" phrases
- `threat_language_count`: "Account will be closed"
- `financial_term_count`: "bank account", "credit card"
- `immediate_action_count`: "Act now", "Don't wait"
- `verification_request_count`: "Verify your account"
- `click_here_count`: "Click here" frequency
- `password_request_count`: Password-related requests
- `account_suspended_count`: Suspension warnings
- `url_in_body_count`: URLs in body text

### Structural Features (`StructuralFeatureExtractor`)

- `html_text_ratio`: HTML to text content ratio
- `has_attachments`: Contains attachments
- `attachment_count`: Number of attachments
- `has_executable_attachment`: Contains .exe, .bat etc.
- `has_office_attachment`: Contains .docx, .xlsx etc.
- `embedded_image_count`: Embedded (base64) images
- `external_image_count`: External image references
- `has_forms`: Contains HTML forms (suspicious)
- `has_javascript`: Contains JavaScript (suspicious)
- `email_size_kb`: Email size in KB

### Linguistic Features (`LinguisticFeatureExtractor`)

- `spelling_error_rate`: Misspelled words ratio
- `grammar_score_proxy`: Grammar quality issues
- `formality_score`: Informal language detection
- `reading_ease_score`: Text readability
- `sentence_count`: Number of sentences
- `avg_sentence_length`: Average sentence length
- `exclamation_mark_count`: Exclamation usage
- `question_mark_count`: Question usage
- `all_caps_ratio`: ALL-CAPS words ratio
- `punctuation_ratio`: Punctuation density

### Financial Features (`FinancialFeatureExtractor`) ⭐

- `bank_impersonation_score`: Bank name similarity (Levenshtein)
- `wire_urgency_score`: Wire transfer urgency
- `credential_harvesting_score`: Credential request patterns
- `invoice_terminology_density`: Invoice/payment terms
- `account_number_request`: Account # requests
- `routing_number_request`: Routing # requests
- `ssn_request`: SSN requests (highly suspicious)
- `payment_urgency_score`: Payment urgency
- `financial_institution_mentions`: Bank mentions
- `wire_transfer_keywords`: Wire transfer terms

## Feature Importance Analysis

```python
from src.analysis.importance import (
    compute_mutual_information,
    rank_features_by_importance,
    print_feature_ranking
)

# Compute mutual information
mi_scores = compute_mutual_information(features, labels)

# Rank top features
ranking = rank_features_by_importance(
    features, labels,
    method='mutual_info',
    top_n=20
)

# Print ranking
print_feature_ranking(ranking, "Top 20 Features - Mutual Information")
```

## Correlation Analysis

```python
from src.analysis.correlation import (
    analyze_feature_correlation,
    remove_redundant_features
)

# Analyze correlations
results = analyze_feature_correlation(
    features,
    threshold=0.9,
    plot=True,
    save_dir="output/"
)

# Remove redundant features
features_reduced, removed = remove_redundant_features(
    features,
    threshold=0.9
)

print(f"Removed {len(removed)} redundant features")
print(f"Remaining: {features_reduced.shape[1]} features")
```

## Custom Pipeline Configuration

```python
from src.transformers import create_custom_pipeline

# Create pipeline with specific extractors
pipeline = create_custom_pipeline(
    include_url=True,
    include_header=True,
    include_sender=True,
    include_content=True,
    include_structural=True,
    include_linguistic=True,
    include_financial=True,  # Always include for banking
    normalize=True
)

# Use pipeline
features = pipeline.fit_transform(emails_df)
```

## Datasets

This pipeline is designed for the following datasets:

1. **Nazario Phishing Corpus**: Public phishing email corpus
2. **APWG eCrime Dataset**: Anti-Phishing Working Group data
3. **Enron Email Dataset**: Legitimate emails (negative class)
4. **Custom Synthetic Banking Phishing**: Financial-specific phishing examples

## Performance

- **Target extraction time**: <100ms per email
- **Features**: 60+ normalized features in [0, 1] range
- **Error handling**: Graceful handling of malformed emails

## Project Structure

```
phishing_email_analysis/
├── src/
│   ├── feature_extractors/    # 7 feature extractors
│   ├── transformers/          # Pipeline and normalization
│   ├── utils/                 # Email parsing, NLP helpers
│   └── analysis/              # Importance, correlation analysis
├── tests/
│   └── test_extractors/       # Unit tests per extractor
├── config/
│   └── banks.json             # Financial institutions list
├── notebooks/                 # EDA, analysis notebooks
└── docs/
    └── FEATURE_CATALOG.md     # Detailed feature documentation
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_extractors/test_financial_features.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Use Case: Federated Learning

This pipeline is designed to feed into a **federated learning system** for financial institutions:

- Each bank trains locally on their email data
- Feature extraction is standardized across institutions
- Privacy-preserving: Only feature updates are shared
- Financial-specific features improve cross-institutional detection

## Contributing

This project is part of a research portfolio on federated phishing detection.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{phishing_email_analysis,
  title={Phishing Email Feature Engineering Pipeline},
  author={Your Name},
  year={2025},
  note={Financial services phishing detection}
}
```

## License

MIT License - See LICENSE file for details.

---
