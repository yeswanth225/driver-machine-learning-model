# Complete Dataset Download Verification Audit — Datasets A to G
**Project**: Intelligent Driver Activity Monitoring & Safety Agent  
**Audit Target Directory**: `D:\driver ml model\driver-monitoring-agent\dataset\raw`  
**Execution Timestamp**: 2026-11-16  
**Auditor**: Senior Machine Learning Engineer & Software Architect

---

## 1. Executive Summary Table

| Dataset | Folder exists | Actual data files | Extracted | Readable | Size | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dataset A1 (Drivers)** | ✅ Yes | 21 CSV | ✅ Yes | ✅ Yes (21/21) | 98.18 MB | **VERIFIED** |
| **Dataset A2 (Daywise)** | ✅ Yes | 48 non-empty (42 CSV + 6 XLSX) | ✅ Yes | ✅ Yes (48/48 non-empty) | 175.39 MB | **VERIFIED** |
| **Dataset B (`dataset_02_jj3tw8kj6h`)** | ✅ Yes | 19 (18 CSV + 1 DOCX) | ✅ Yes | ✅ Yes (19/19) | 19.92 MB | **VERIFIED** |
| **Dataset C (`dataset_03_5stn873wft`)** | ✅ Yes | 3 CSV | ✅ Yes | ✅ Yes (3/3) | 2.82 MB | **VERIFIED** |
| **Dataset D (`dataset_04_eishkaran`)** | ✅ Yes | 78 (72 CSV) | ✅ Yes | ✅ Yes (72/72) | 250.29 MB | **VERIFIED** |
| **Dataset E (`dataset_05_shashwatwork`)** | ✅ Yes | 18 CSV | ✅ Yes | ✅ Yes (18/18) | 19.88 MB | **VERIFIED** |
| **Dataset F (`dataset_06_yotamkatz`)** | ✅ Yes | 7 (5 CSV + 1 MD + 1 PNG) | ✅ Yes | ✅ Yes (7/7) | 11.96 MB | **VERIFIED** |
| **Dataset G (`dataset_07_vitorrf`)** | ✅ Yes | 238 (237 CSV + 1 RTF) | ✅ Yes | ✅ Yes (238/238) | 2,265.62 MB | **VERIFIED** |

---

## 2. Global Quantitative Metrics

- **Total Datasets Inspected**: All 7 datasets (A through G).
- **Total Datasets Fully Verified**: **All 7 datasets (100% complete)**.
- **Total Missing Datasets**: **0 datasets**.
- **Total Actual Extracted Data Files**: **432 files** across all datasets.
- **Total Raw Dataset Storage Size**: **2.84 GB** (2,844.06 MB / 2,982,217,357 bytes).
- **Total Validated Records / Rows across all CSVs**: **40,011,179 rows**.

---

## 3. Dataset Breakdown & Verification

### 3.1 Dataset A (Drivers & Daywise)
- **Status:** **VERIFIED**
- **Files:** 21 CSV in Drivers (1,826,637 rows), 56 CSV + 6 XLSX in Daywise (2,885,476 rows across 42 non-empty CSVs; 14 empty GPS CSVs).
- **Size:** 273.57 MB combined.

### 3.2 Dataset B (Raspberry Pi Driving Behavior)
- **Status:** **VERIFIED**
- **Files:** 18 CSV (19,848 rows: 1,114 raw + 18,734 window features) + 1 DOCX.
- **Size:** 19.92 MB.
- **Anchor SHA-256 (`sensor_raw.csv`):** `9b3a0667562e57eaf5cd7e0c656349518bebf7095dac4627f2353051858d1594`.

### 3.3 Dataset C (Phone Sensor Data While Driving)
- **Status:** **VERIFIED**
- **Files:** 3 CSV (28,498 rows: 14,249 labeled + 14,249 raw trip records).
- **Size:** 2.82 MB.
- **Anchor SHA-256 (`3_FinalDatasetCsv.csv`):** `47f2065e43f02d7721b96efa1052cbdeeee098730f004423a153232523d7f8aa`.

### 3.4 Dataset D (`dataset_04_eishkaran`)
- **Status:** **VERIFIED**
- **Source:** Extracted from `archive (1).zip` (SHA-256: `eb134eb1643976a9914641b7709a61b97b20c7a54b84dad4397e9796ab9173a5`).
- **Files:** 78 files (72 CSVs across 8 session folders: 7 timestamped driver folders + `SensorRecord2020`).
- **Total Rows:** **4,173,442 rows**.
- **Size:** 250.29 MB.
- **Identity Distinction vs Dataset A:** Folders are named by timestamp rather than Driver/Daywise; contains 72 CSVs and 4,173,442 rows compared to Dataset A's 77 CSVs and 4,712,113 rows. Proved to be a non-identical variant.

### 3.5 Dataset E (`dataset_05_shashwatwork`)
- **Status:** **VERIFIED**
- **Source:** Extracted from `archive (2).zip` (SHA-256: `360b060f00339cf7a495c6a809457753e40eba6af65e6abcc603bfe186134d3c`).
- **Files:** 18 CSVs (`features_14.csv`, `sensor_raw.csv`, and 16 `sero_features_*.csv`).
- **Total Rows:** **19,848 rows**.
- **Size:** 19.88 MB.
- **Anchor SHA-256 (`sensor_raw.csv`):** `a94038fc06b2e6f1bd909dd6f9bbd94520913483eb547dc4be0aafa6922f4831`.
- **Identity Distinction vs Dataset B:** In Dataset E, `sensor_raw.csv` has 7 columns (renamed `Target(Class)`), whereas Dataset B has 12 columns (including `DriverID`, `TaskID`, and timestamp components). Proved to be a non-identical processed variant.

### 3.6 Dataset F (`dataset_06_yotamkatz`)
- **Status:** **VERIFIED**
- **Source:** Extracted from `archive (3).zip` (SHA-256: `b511fc8d623b4ede60912d6fe463361d853cdf3aec28cf072ed8e38b017dd728`).
- **Files:** 7 files (`README.md`, `all_drivers.csv`, `driver_E.csv`, `driver_G.csv`, `driver_R.csv`, `driver_Y.csv`, `route_overview.png`).
- **Total Rows:** **82,038 rows** (41,019 master + 41,019 in driver files).
- **Columns (21):** OBD/telematics features (`driver`, `elapsed_s`, `Calculated load value (%)`, `Engine RPM (RPM)`, `Vehicle speed (km/h)`, etc.).
- **Size:** 11.96 MB.
- **Anchor SHA-256 (`all_drivers.csv`):** `6bf5684b7bbd36ee504a49304fb63a59856dbe7e5b874167a2132d84233416af`.

### 3.7 Dataset G (`dataset_07_vitorrf`)
- **Status:** **VERIFIED**
- **Source:** Extracted from `archive (4).zip` (SHA-256: `0329b970dec824f84993a7fc1918c8bf85fb1fa11720f88288370b45528a73d1`).
- **Files:** 238 files (237 CSV files + 1 `DataInfo.rtf`).
- **Total Rows:** **30,995,196 rows**.
- **Size:** 2,265.62 MB (2.21 GB).
- **Subdirectories:** `Processed Data/` (39 files), `RawData/` (77 files), `TripData/` (nested original collection).
- **License:** Open Database License (ODbL) / DbCL.

---

## 4. Git & GitHub Storage Policy for Large Datasets

With Dataset G adding 2.21 GB of data:
- **Files > 100 MB:** None (largest CSV files in Dataset G are ~22 MB).
- **Total Repository Size:** ~2.84 GB.
- **Recommendation:**
  - For standard Git pushing to free GitHub accounts (recommended limit 2 GB, hard limit 5 GB):
  - Datasets A, B, C, D, E, F (totaling ~575 MB) can be committed directly to Git without reaching repository limits.
  - Dataset G (2.21 GB) should be tracked using **Git LFS** or stored as a companion download to respect GitHub's standard repo quotas, OR the entire repository can be tracked with Git LFS for all `dataset/raw/**` CSV files.
  - `.gitattributes` has been configured accordingly.
