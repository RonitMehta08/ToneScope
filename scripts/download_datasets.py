"""
Dataset download script.

Downloads all required datasets for ToneScope training and evaluation.
All datasets are real, publicly available, and peer-reviewed.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_all_datasets() -> None:
    """Download and cache all required datasets via HuggingFace Hub."""
    from datasets import load_dataset

    print("=" * 60)
    print("ToneScope — Dataset Download")
    print("=" * 60)

    datasets_to_download = [
        {
            "name": "GoEmotions",
            "path": "google-research-datasets/go_emotions",
            "config": "simplified",
            "splits": ["train", "validation", "test"],
        },
        {
            "name": "MELD",
            "path": "declare-lab/MELD",
            "config": None,
            "splits": ["train", "validation", "test"],
        },
        {
            "name": "DailyDialog",
            "path": "daily_dialog",
            "config": None,
            "splits": ["train", "validation", "test"],
        },
        {
            "name": "iSarcasmEval",
            "path": "Ammar-alhaj-ali/iSarcasmEval",
            "config": None,
            "splits": ["train", "test"],
        },
    ]

    for ds_info in datasets_to_download:
        name = ds_info["name"]
        print(f"\n{'─' * 40}")
        print(f"Downloading: {name}")
        print(f"  Source: {ds_info['path']}")

        try:
            for split in ds_info["splits"]:
                kwargs = {"split": split}
                if ds_info["config"]:
                    kwargs["name"] = ds_info["config"]

                ds = load_dataset(ds_info["path"], **kwargs)
                print(f"  ✓ {split}: {len(ds)} samples")

            print(f"  ✓ {name} downloaded successfully!")

        except Exception as e:
            print(f"  ✗ Failed to download {name}: {e}")
            print(f"  → This dataset may require manual download or an alternate source.")

    # Download the base model tokenizer/weights
    print(f"\n{'─' * 40}")
    print("Downloading: roberta-base model weights")
    try:
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = AutoModel.from_pretrained("roberta-base")
        print("  ✓ roberta-base downloaded and cached!")
        del model, tokenizer
    except Exception as e:
        print(f"  ✗ Failed to download roberta-base: {e}")

    print(f"\n{'=' * 60}")
    print("Download complete! Datasets are cached in your HuggingFace cache directory.")
    print("=" * 60)


if __name__ == "__main__":
    download_all_datasets()
