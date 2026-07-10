"""
Task 20 — Deliverable Packager
PlaceMux Phase 1 Industry Immersion

Produces `placemux_task20_YYYYMMDD.zip` containing all Task 20 files,
the serialized model artifact, and logs, as required by the standing rules.
"""

import os
import zipfile
import datetime

def main():
    task_num = 20
    date_str = datetime.date.today().strftime("%Y%m%d")
    zip_name = f"placemux_task{task_num:02d}_{date_str}.zip"
    
    # Files/directories to include in the ZIP
    targets = {
        "src/serve_task20.py": "src/serve_task20.py",
        "src/test_task20.py": "src/test_task20.py",
        "src/zip_task20.py": "src/zip_task20.py",
        "run_task20.bat": "run_task20.bat",
        "requirements.txt": "requirements.txt",
        "models/task19/placemux_pipeline_v1.0.0.joblib": "models/task19/placemux_pipeline_v1.0.0.joblib",
        "models/task19/metadata.json": "models/task19/metadata.json",
        "logs/task20.log": "logs/task20.log",
        "logs/task20_test_results.json": "logs/task20_test_results.json",
        "logs/task20_walkthrough.md": "logs/task20_walkthrough.md"
    }

    print(f"Packaging deliverables for Task {task_num}...")
    
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for disk_path, zip_path in targets.items():
            if os.path.exists(disk_path):
                zf.write(disk_path, arcname=zip_path)
                print(f"  Added: {zip_path}")
            else:
                print(f"  [WARNING] File not found, skipping: {disk_path}")
                
    print(f"\n[OK] ZIP created successfully: {zip_name}")

if __name__ == "__main__":
    main()
