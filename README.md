$readmeUpdate = @"
# 🏦 CrediTrust: Intelligent Complaint Analysis (Production-Ready)

## 📊 Technical Standards
- **Stratification:** Exactly 3,000 samples per product category.
- **Chunking:** 500 characters with 50-character overlap.
- **Automation:** GitHub Actions CI enabled; Unit tests in ``/tests``.

## 📂 Configurable Parameters
All system settings are now centralized in ``src/config.py`` to prevent documentation mismatches.

## 🛠️ How to Run
1. Place raw data in ``data/raw/complaints.csv``
2. Run Preprocessing Notebook.
3. Execute Embedding: ``python -m src.embedding_pipeline``
4. Run Tests: ``python -m unittest discover tests``
"@
$readmeUpdate | Out-File -FilePath "README.md" -Encoding utf8