# Dataset Overview: Driver Behavior Detection Using Smartphone

## 1. Provenance and Acquisition Details

| Attribute | Verified Specification |
| :--- | :--- |
| **Dataset Title** | Driver Behavior Detection Using Smartphone - Dataset |
| **Principal Investigators** | Pawan S. Wawage & Dr. Yogesh D. Deshpande (Vishwakarma University, Pune, India) |
| **Primary Publication** | *"Smartphone Sensor Dataset for Driver Behavior Analysis"*, *Data in Brief*, Vol. 41, 107992 (Elsevier, 2022) |
| **Identifiers** | DOI: [10.17632/9vr83n7z5j.2](https://doi.org/10.17632/9vr83n7z5j.2) \| PMCID: PMC8914310 \| PMID: 35282175 |
| **Collection Medium** | Android application (*Sensor Record*) installed on Xiaomi Redmi 4 and Mi A3 smartphones |
| **Mounting Position** | Placed horizontally inside vehicle utility box / dashboard compartment |
| **Vehicle Types** | Light Motor Vehicles (LMVs): Ford Figo 1.2, Maruti Suzuki Swift VXI, Tata Nexon XMS |
| **Subjects & Cohorts** | Drivers aged 30–35 with 5+ years experience driving in real-world Indian road traffic |
| **Trip Characteristics** | Naturalistic routes comprising city traffic, arterial roads, and highway segments (10–25 km per trip) |

---

## 2. Directory Structure and Cohort Taxonomy

The dataset is partitioned into two distinct experimental collections in `driver-monitoring-agent/dataset/raw/`:

```text
dataset/raw/
├── drivers/                         # Multi-Subject Collection (Table 3 in publication)
│   ├── Driver-1/                    # Data collected across 7 distinct human drivers
│   │   ├── Accelerometer.csv        # 3-axis linear acceleration
│   │   ├── Gyroscope.csv            # 3-axis rotational angular velocity
│   │   └── Pressure.csv             # Barometric atmospheric pressure
│   ├── Driver-2/ ... Driver-7/      # Identical structure per driver
└── daywise/                         # Single-Subject Multi-Day Collection (Table 4 in publication)
    └── Daywise data/                # Data collected for a specific single driver across 7 days
        ├── Day-1S/ & Day-1R/        # S = Outbound Trip (Start); R = Return Trip
        ├── Day-2S/ & Day-2R/
        ├── Day-3S/, Day-3R/, Day-3E/# E = Evening / Extra Trip
        ├── Day-4S/, Day-4R/, Day-4E1/, Day-4E2/, Day-4E3/
        ├── Day-5S/
        ├── Day-6S/ & Day-6R/
        └── Day-7S/
```

> [!NOTE]
> **Cohort Relationship**: As documented in PMC8914310, `drivers/` contains recordings across **7 distinct drivers** (Table 3), while `daywise/` contains multi-day trips for a **single specific driver** (Table 4).

---

## 3. Sensor Features, Units, and Orientation Limitations

| Sensor Channel | Data File | Recorded Axes / Columns | Verified Units | Physical Meaning & Nominal Ranges |
| :--- | :--- | :--- | :--- | :--- |
| **Accelerometer** | `Accelerometer.csv` | `Timestamp, Unix Timestamp, Milliseconds, X, Y, Z` | $\text{m/s}^2$ | 3-axis linear acceleration including Earth gravity vector. Vector norm $\\|a\\| = \sqrt{X^2+Y^2+Z^2} \approx 9.81\ \text{m/s}^2$. |
| **Gyroscope** | `Gyroscope.csv` | `Timestamp, Unix Timestamp, Milliseconds, X, Y, Z` | $\text{rad/s}$ | 3-axis angular velocity (Android `Sensor.TYPE_GYROSCOPE` format). Centered at $0.00 \pm 0.05\ \text{rad/s}$ at constant heading; turns reach $0.3 - 1.5\ \text{rad/s}$. *(Note: Table 2 caption casually wrote "$^\circ/\text{s}$", but numerical magnitudes are $\text{rad/s}$).* |
| **Pressure** | `Pressure.csv` | `Timestamp, Unix Timestamp, Milliseconds, Millibars` | $\text{mbar}$ ($\text{hPa}$) | Atmospheric pressure measured by smartphone barometer. Nominal range: $940 - 970\ \text{hPa}$. Sampled at 10 Hz. |
| **Proximity** | `Proximity.csv` | `Timestamp, Milliseconds, centimeters` | $\text{cm}$ | Infrared proximity sensor. Sampled at 10 Hz ($5.0\ \text{cm}$ unblocked vs $0.0\ \text{cm}$ occluded). |
| **GPS** | `GPS.csv` | *(Empty / 0 bytes in all recordings)* | N/A | Deactivated during logging. **No GPS speed or coordinate data is available**. |
| **Rotation Vector** | `RotationVector.csv` | `Timestamp, Milliseconds, X, Y, Z, cos, headingAccuracy` | Unitless | Orientation quaternion (only present in aborted session `Day-4E1`, 1 row). |

> [!WARNING]
> **Smartphone Orientation Limitation**: The phone was placed "horizontally in the vehicle utility box", but vehicle-relative heading alignment was not calibrated. Direct mapping of $X$ to lateral and $Y$ to longitudinal is uncalibrated. Euclidean vector norms and orientation-invariant features are recommended.

---

## 4. Audited Record Counts and Statistics

### Multi-Subject Collection (`drivers/` - Table 3)

| Driver Identifier | Accelerometer Rows | Gyroscope Rows | Pressure Rows | Verified Subtotal | Approx. Duration |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Driver-1** | 100,199 | 100,748 | 10,133 | 211,080 | ~16.7 min |
| **Driver-2** | 1,212 | 1,224 | 124 | 2,560 | ~0.2 min (short test) |
| **Driver-3** | 106,958 | 107,551 | 107,551 | 322,060 | ~17.8 min |
| **Driver-4** | 177,277 | 178,271 | 178,271 | 533,819 | ~29.5 min |
| **Driver-5** | 93,325 | 93,857 | 93,857 | 281,039 | ~15.5 min |
| **Driver-6** | 54,819 | 55,133 | 55,133 | 165,085 | ~9.1 min |
| **Driver-7** | 103,276 | 103,859 | 103,859 | 310,994 | ~17.2 min |
| **Drivers Subtotal** | **637,066** | **640,643** | **548,928** | **1,826,637 rows** | **~106 min total** |

### Single-Driver Multi-Day Collection (`daywise/` - Table 4)

| Sensor Channel | File Count | Row Count | Subtotal Bytes | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Accelerometer** | 16 | 1,250,536 | 71.65 MB | Complete across all 16 sessions |
| **Gyroscope** | 14 | 1,281,862 | 80.71 MB | Missing in Day-2R and Day-6R |
| **Proximity** | 11 | 353,077 | 12.98 MB | Intermittently recorded |
| **RotationVector** | 1 | 1 | 91 bytes | Only in aborted Day-4E1 session |
| **GPS** | 14 | 0 | 0 bytes | All 14 files are 0 bytes |
| **Daywise Subtotal** | **56 CSVs** | **2,885,476 rows** | **165.34 MB** | **Single driver across 7 days** |

### Verified Dataset Grand Total
$$\text{Total Records} = 1,826,637 \ (\text{drivers}) + 2,885,476 \ (\text{daywise}) = \mathbf{4,712,113\ \text{rows across 77 CSV files}}$$

---

## 5. Audited Sampling Rates

| Sensor Modality | Median Time Delta ($\Delta t$) | Effective Sampling Frequency | Jitter Range |
| :--- | :---: | :---: | :---: |
| **Accelerometer** | $10.0\ \text{ms}$ | **$100.0\ \text{Hz}$** | $9 - 12\ \text{ms}$ |
| **Gyroscope** | $10.0\ \text{ms}$ | **$100.0\ \text{Hz}$** | $9 - 12\ \text{ms}$ |
| **Pressure** | $100.0\ \text{ms}$ | **$10.0\ \text{Hz}$** | $99 - 101\ \text{ms}$ |
| **Proximity** | $100.0\ \text{ms}$ | **$10.0\ \text{Hz}$** | $95 - 102\ \text{ms}$ |
