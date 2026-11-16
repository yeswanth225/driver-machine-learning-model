# DATASET_AUDIT_REPORT.md
**Project**: Intelligent Driver Activity Monitoring & Safety Agent  
**Version**: 3.0 (Datasets A through G Complete Verification)  
**Updated**: 2026-11-16  
**Status**: ✅ All 7 Datasets Verified and Organized

---

## 1. Available Datasets Summary

| ID | Name / Slug | Source | DOI / Reference | Folder | Status | Total Files | Total Rows | Total Size |
|:---|:---|:---|:---|:---|:---:|:---:|:---:|:---:|
| **A1** | Smartphone Driving (Drivers) | Mendeley Data | 10.17632/9vr83n7z5j.2 | `raw/drivers/` | ✅ **VERIFIED** | 21 CSV | 1,826,637 | 98.18 MB |
| **A2** | Smartphone Driving (Daywise) | Mendeley Data | 10.17632/9vr83n7z5j.2 | `raw/daywise/` | ✅ **VERIFIED** | 56 CSV + 6 XLSX | 2,885,476 | 175.39 MB |
| **B** | Driving Behavior (Raspberry Pi) | Mendeley Data | 10.17632/jj3tw8kj6h.3 | `raw/dataset_02_jj3tw8kj6h/` | ✅ **VERIFIED** | 18 CSV + 1 DOCX | 19,848 | 19.92 MB |
| **C** | Phone Sensor Data While Driving | Mendeley Data | 10.17632/5stn873wft.1 | `raw/dataset_03_5stn873wft/` | ✅ **VERIFIED** | 3 CSV | 28,498 | 2.82 MB |
| **D** | Driver Behaviour Analysis | Kaggle | `eishkaran/...` | `raw/dataset_04_eishkaran/` | ✅ **VERIFIED** | 78 files (72 CSV) | 4,173,442 | 250.29 MB |
| **E** | Driving Behavior Dataset | Kaggle | `shashwatwork/...` | `raw/dataset_05_shashwatwork/` | ✅ **VERIFIED** | 18 CSV | 19,848 | 19.88 MB |
| **F** | Car Driver Behavior (4 Drivers) | Kaggle | `yotamkatz/...` | `raw/dataset_06_yotamkatz/` | ✅ **VERIFIED** | 7 files (5 CSV) | 82,038 | 11.96 MB |
| **G** | Car Trips Data Log / Mining | Kaggle | `vitorrf/...` | `raw/dataset_07_vitorrf/` | ✅ **VERIFIED** | 238 files (229 CSV + 9 RTF/meta) | 30,995,196 | 2,265.62 MB |

---

## 2. Dataset Attribution and Academic Lineage

Full details are documented in [`docs/DATASET_LICENSES_AND_ATTRIBUTION.md`](../docs/DATASET_LICENSES_AND_ATTRIBUTION.md).
All datasets are governed by open research licenses (**CC BY 4.0**, **CC BY-SA 4.0**, and **ODbL/DbCL**), granting rights for research, adaptation, and redistribution with attribution.

---

## 3. Storage and GitHub Distribution

- **Total Repository Dataset Size**: **2.84 GB** across 432 data files.
- **Total Validated Records**: **40,011,179 rows**.
- **Portability**: All datasets are self-contained within `dataset/raw/` and can be checked using [`ml/preprocessing/verify_dataset_integrity.py`](../ml/preprocessing/verify_dataset_integrity.py).
