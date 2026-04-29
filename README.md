# Digital Phenotyping of Traditional Chinese Medicine Constitution

This repository contains reproducible code and public aggregate outputs for the manuscript:

**Longitudinal digital phenotyping of Traditional Chinese Medicine constitution using routine health examination data**

The study develops a longitudinal digital phenotyping framework for Traditional Chinese Medicine (TCM) constitution using routine health examination records. The public repository is designed to support transparency while protecting participant privacy.

## Repository Contents

- `scripts/`: analysis, figure-generation, PLOS asset packaging, and DOCX-building scripts.
- `outputs/plos_submission/`: PLOS-oriented main tables, figure files, supporting information, and submission-package README.
- `outputs/initial_analysis/tables/`: public aggregate tables required for figures and manuscript tables.
- `outputs/advanced_analysis/tables/`: public aggregate tables from supplementary analyses.
- `outputs/manuscript/`: manuscript drafts, cover letter draft, and submission strategy documents.
- `data/synthetic/`: synthetic demonstration data for code demonstration only.
- `data/`: restricted-data instructions and data dictionary files.
- `metadata/`: public file manifest and restricted-file exclusion list.

## Data Availability

The original participant-level routine health examination dataset is not included in this repository because it contains sensitive health information from older adults. Qualified researchers may request access to the de-identified analytic dataset from the corresponding author:

Min Chen  
Email: 150325684@qq.com

Data access is subject to institutional approval, data-use agreement, ethics approval where applicable, and relevant privacy regulations.

## What Can Be Reproduced Publicly

Without restricted participant-level data, users can:

1. Inspect all public aggregate result tables used in the manuscript.
2. Recreate PLOS-oriented submission tables from public aggregate outputs.
3. Inspect publication figures and figure file indices.
4. Review all analysis scripts used to produce the manuscript.
5. Run demonstration code on synthetic data to understand the expected feature schema.

Full re-analysis from raw participant-level records requires restricted data access.

## Quick Start

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run the synthetic-data demonstration:

```bash
python scripts/run_synthetic_demo.py
```

Regenerate PLOS-oriented tables and figure-file indices from included aggregate outputs:

```bash
python scripts/create_plos_submission_assets.py
```

Build the manuscript DOCX from the Markdown draft and generated tables:

```bash
python scripts/build_plos_docx.py
```

Regenerate publication figures from included aggregate analysis tables:

```bash
python scripts/create_publication_figures.py
```

The synthetic demo is intentionally simple. It confirms that the public feature schema can be loaded and that a constitution-screening workflow can be executed without access to participant-level data. It does not reproduce the manuscript results.

## Restricted Full Analysis

The full raw-data analysis script is:

```bash
python scripts/initial_constitution_analysis.py
```

This script requires the restricted source workbook and therefore is not runnable from the public repository alone. Place restricted data only in a private, non-version-controlled location. Do not commit raw participant-level data.

## Privacy Protection

The following files were intentionally excluded from this public package:

- Raw Excel workbook
- Participant-level baseline analysis frame
- Adjacent visit-pair long table
- Participant-level constitution-transition long table
- Pickled model frame

See `metadata/excluded_sensitive_files.csv` for the explicit exclusion list.

## Citation

If this repository supports your work, please cite the associated manuscript after publication. A provisional `CITATION.cff` file is included for GitHub citation metadata and should be updated with DOI information after publication.

## License

Code is released under the MIT License. Aggregate tables, documentation, and figure files are released under CC BY 4.0 unless a journal or institutional policy requires a different final setting.
