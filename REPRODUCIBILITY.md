# Reproducibility Notes

This repository supports two levels of reproducibility.

## Public Reproduction

The public repository includes aggregate tables, figures, manuscript drafts, analysis scripts, and synthetic demonstration data. With these files, users can inspect the reported results, regenerate PLOS-oriented tables and figure indices, rebuild the manuscript DOCX, and run a synthetic-data demonstration of the model pipeline.

## Restricted Full Re-analysis

Full re-analysis from raw participant-level records requires controlled access to the de-identified analytic dataset. The source workbook and participant-level derived tables are intentionally excluded from the repository. Qualified researchers may request access from the corresponding author, subject to institutional approval, data-use agreement, ethics approval where applicable, and privacy regulations.

## Expected Workflow

1. Install dependencies with `pip install -r requirements.txt`.
2. Run `python scripts/run_synthetic_demo.py` to verify the public environment.
3. Run `python scripts/create_plos_submission_assets.py` to regenerate submission table and figure indices from aggregate outputs.
4. Run `python scripts/create_publication_figures.py` to regenerate publication figures from public aggregate tables.
5. Run `python scripts/build_plos_docx.py` to rebuild the manuscript DOCX from the Markdown draft and generated tables.

Scripts that begin with raw data cleaning require restricted files and are documented as restricted workflows.
