from huggingface_hub import hf_hub_download
from pathlib import Path

SUBSAMPLE = "up0.001_ip0.001"
LOCAL_DIR = Path("data")

files = [
    f"subsamples/{SUBSAMPLE}/train/week_00.parquet",
    f"subsamples/{SUBSAMPLE}/train/week_24.parquet",
    f"subsamples/{SUBSAMPLE}/validation/week_25.parquet",
    "metadata/items_metadata.parquet",
]

for file in files:
    print(f"Скачиваю {file}...")
    hf_hub_download(
        repo_id="deepvk/VK-LSVD",
        repo_type="dataset",
        filename=file,
        local_dir=str(LOCAL_DIR),
    )

print("Готово! Данные скачаны в папку data/")