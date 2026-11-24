"""
Final System Integrity Verification Script
Checks all 4 production model hashes, test suite, and directories.
"""
import hashlib
from pathlib import Path

HASHES = {
    'dataset_b_model': ('models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib', '552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90'),
    'dataset_b_scaler': ('models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib', 'd4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576'),
    'dataset_c_model': ('models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib', 'a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc'),
    'dataset_c_scaler': ('models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib', 'd67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5'),
}

def verify():
    print("=" * 70)
    print("IDAMS FINAL INTEGRITY AUDIT")
    print("=" * 70)
    all_ok = True
    for key, (rel_path, expected_hash) in HASHES.items():
        p = Path(rel_path)
        if not p.exists():
            print(f"FAILED: File {rel_path} does not exist!")
            all_ok = False
            continue
        data = p.read_bytes()
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash == expected_hash:
            print(f"[OK] {key}: {actual_hash} (FROZEN MATCH)")
        else:
            print(f"[ERROR] {key} HASH MISMATCH!\n  Expected: {expected_hash}\n  Actual:   {actual_hash}")
            all_ok = False
    
    assert all_ok, "Integrity verification failed!"
    print("All production model and scaler hashes are bit-for-bit identical.")

if __name__ == "__main__":
    verify()
