from datasets import load_dataset
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN") or os.getenv("GAIA_HUGGINGFACE_API_KEY")

print(f"Token present: {bool(token)}")

ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation", token=token)
# Look for the calculus task
task_id = "1f975693-876d-457b-a649-393859e79bf3"
row = next((r for r in ds if r["task_id"] == task_id), None)

if row:
    print(f"Task found: {row['Question'][:50]}...")
    print(f"File Name: {row.get('file_name')}")
    print(f"File Path: {row.get('file_path')}")
    if row.get('file_path'):
        print(f"File Path Exists? {os.path.exists(row['file_path'])}")
else:
    print("Task not found in dataset.")
