import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.build_features import build_dataset

build_dataset(
    audio_directory="data/raw/fma_small/000",
    output_path="data/processed/music_features.parquet"
)