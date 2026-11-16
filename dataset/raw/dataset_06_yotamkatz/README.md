# Driver Behavior — OBD-II + Smartphone Sensors (Same-Car, Same-Route)

A small, clean dataset for **behavioral driver identification**: four different drivers each
drove the **same vehicle** over the **same route** on the **same evening**, while a smartphone
logged on-board diagnostics (OBD-II), inertial sensors (IMU), and GPS at roughly **7–11 Hz**.

Because everyone drove an identical car and route under similar conditions, differences between
the trips come primarily from **how each person drives** — making this a focused testbed for
driver-fingerprinting / telematics models without route or vehicle confounders.

## Files

| File | Driver | Rows | Duration kept |
|------|--------|------|---------------|
| `driver_Y.csv` | Y | 12,582 | 19.1 min |
| `driver_R.csv` | R | 10,103 | 15.4 min |
| `driver_E.csv` | E |  7,442 | 18.5 min |
| `driver_G.csv` | G | 10,892 | 18.3 min |
| `all_drivers.csv` | all | 41,019 | all four trips stacked |
| `route_overview.png` | — | — | Per-driver route maps, colored by vehicle speed |

Each `driver_*.csv` is one continuous trip; `all_drivers.csv` is the four concatenated into a
single long-format table (use the `driver` column to separate them). The four drivers are
anonymized as **Y, R, E, G**.

## Privacy / preprocessing note

The trips were round-trips that started and ended at a private residence. To protect that
address, **all samples within ~300 m of the start/end point were removed** from each trip
(roughly the first 1–2 minutes and last 2–3 minutes). Every published GPS sample is therefore
**at least ~300 m from home**, and the original route map (which contained a pin on the
residence) has been replaced with a coordinate-free `route_overview.png` generated from the
already-trimmed tracks. The absolute timestamp column was also replaced with a per-trip
**`elapsed_s`** (seconds from each trip's first sample), removing the recording date and
time-of-day. No sensor values were otherwise altered.

## Data dictionary

| Column | Units | Source | Description |
|--------|-------|--------|-------------|
| `driver` | — | label | Anonymized driver id: `Y`, `R`, `E`, `G` |
| `elapsed_s` | s | derived | Seconds since each trip's first (post-trim) sample; resets to 0 per driver |
| `Calculated load value` | % | OBD-II | Engine load |
| `Engine coolant temperature` | °C | OBD-II | Coolant temp |
| `Engine RPM` | RPM | OBD-II | Engine speed |
| `Vehicle speed` | km/h | OBD-II | Speed reported by the ECU |
| `Intake air temperature` | °C | OBD-II | Intake air temp |
| `Mass air flow rate` | g/s | OBD-II | MAF sensor |
| `Absolute throttle position` | % | OBD-II | Throttle position |
| `Control module voltage` | V | OBD-II | ECU module voltage |
| `Latitude` | deg | phone GPS | WGS84 latitude (trimmed near home; `0` = no fix) |
| `Longitude` | deg | phone GPS | WGS84 longitude (trimmed near home; `0` = no fix) |
| `GPS Speed` | km/h | phone GPS | Speed from GPS |
| `Horz Accuracy` | m | phone GPS | Horizontal accuracy estimate |
| `Accel X/Y/Z` | m/s² | phone IMU | Linear acceleration, 3 axes |
| `Rotation Rate X/Y/Z` | deg/s | phone IMU | Gyroscope angular rate, 3 axes |
| `Adapter voltage` | V | OBD adapter | Bluetooth OBD adapter supply voltage |

> Sampling is event-driven, so OBD, GPS and IMU values update at slightly different cadences;
> rows reflect the raw logged stream (not resampled to a fixed grid).

## Suggested tasks

- **Closed-set driver identification** — classify the driver (Y/R/E/G) from short windows.
- **Open-set / "unfamiliar driver" detection** — train on a subset of drivers, flag the rest.
- **Feature engineering for telematics** — jerk, harsh-event rates, RPM/throttle dynamics, etc.

A simple, leakage-aware protocol: split **at the trip/time level** using `elapsed_s` (e.g., train
on the first part of each trip, test on the last part). Use `elapsed_s` only to order/window
samples — **do not** feed `Latitude`, `Longitude`, or `elapsed_s` into the model as features;
they encode the route/timeline and would leak the label.

## Honest limitations

- **One vehicle, one route, one evening, four drivers.** Great for controlled comparison, but
  results will not automatically generalize across cars, routes, or conditions.
- **No ground-truth "theft" labels** — this supports driver-recognition research, not verified
  anomaly/theft detection.
- Small N (4 drivers); treat any accuracy figure with wide uncertainty.

## Attribution & license

Collected as part of an academic capstone on behavioral driver identification from everyday
vehicle data. *(Add your name(s)/affiliation and a Kaggle license — e.g. CC BY 4.0 — here.)*
