import argparse
import csv
import os

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download audio-transcript dataset from Hugging Face."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the Hugging Face dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset",
        help="Directory to save the dataset.",
    )
    parser.add_argument(
        "--audio_column",
        type=str,
        default="audio_file",
        help="The column containing audio.",
    )
    parser.add_argument(
        "--transcript_column",
        type=str,
        default="text",
        help="The column containing transcripts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = os.path.abspath(args.output_dir)
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    dataset = load_dataset(args.dataset_name, split="train")
    metadata_path = os.path.join(output_dir, "metadata.csv")

    with open(metadata_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text"])
        for index, example in enumerate(tqdm(dataset, total=len(dataset))):
            audio = example[args.audio_column]
            text = example[args.transcript_column]

            filename = f"{index:06d}.wav"
            audio_path = os.path.join(audio_dir, filename)

            sf.write(
                audio_path,
                audio["array"],
                audio["sampling_rate"],
            )

            writer.writerow([
                os.path.join(args.output_dir, os.path.relpath(audio_path, output_dir)),
                text.strip(),
            ])

if __name__ == "__main__":
    main()
