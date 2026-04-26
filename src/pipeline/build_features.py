import os
import pandas as pd

from features.feature_extractor import extract_features


def build_dataset(audio_directory, output_path):

    dataset = []

    SUPPORTED_FORMATS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

    for file in os.listdir(audio_directory):

        if not file.lower().endswith(SUPPORTED_FORMATS):
            continue

        audio_path = os.path.join(audio_directory, file)

        print(f"Processing {file}")

        features = extract_features(audio_path)

        features["track_id"] = file

        dataset.append(features)

    df = pd.DataFrame(dataset)

    df.to_parquet(output_path)

    print("Dataset saved to:", output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract audio features from a folder of songs."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to folder containing audio files"
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to save the output Parquet file e.g. output/dataset.parquet"
    )

    args = parser.parse_args()

    build_dataset(args.input_dir, args.output_path)