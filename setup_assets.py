import os
import shutil

# Automated asset copy script for Accivision AI GitHub preview
src = r"C:\Users\Sandheep\.gemini\antigravity\brain\68caabeb-cc88-49cf-9298-00890dc53d85\dashboard_preview_1779029415538.png"
dest_dir = r"e:\Accivision\assets"
dest_file = os.path.join(dest_dir, "dashboard_preview.png")

try:
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.exists(src):
        shutil.copy(src, dest_file)
        print("\n==================================================")
        print(f"[SUCCESS] Dynamic preview banner compiled successfully!")
        print(f"Destination: {dest_file}")
        print("==================================================\n")
    else:
        print(f"\n[ERROR] Source mockup image not found at:\n{src}\n")
except Exception as e:
    print(f"\n[ERROR] Failed to compile project assets: {e}\n")
