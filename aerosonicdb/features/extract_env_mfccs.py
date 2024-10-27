#!/usr/bin/env python
"""Feature extraction logic and entry point for environmental noise dataset."""
import json
import math
import os

import click
import librosa
import pandas as pd
from tqdm.auto import tqdm

from aerosonicdb.utils import get_project_root

ROOT_PATH = get_project_root()

# set the i/o paths
DATASET_PATH = os.path.join(ROOT_PATH, "data/raw")
ENV_AUDIO_PATH = os.path.join(DATASET_PATH, "env_audio")
OUTPUT_PATH = os.path.join(ROOT_PATH, "data/processed")
CLASS_PATH = os.path.join(DATASET_PATH, "environment_class_mappings.csv")

# set the constants and derivatives
SAMPLE_RATE = 22050
DURATION = 5
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

SAMPLES_PER_SEGMENT = SAMPLE_RATE * DURATION
EXPECTED_MFCC_VECTORS_PER_SEGMENT = math.ceil(SAMPLES_PER_SEGMENT / HOP_LENGTH)


def save_env_mfccs(
    env_n,
    duration=DURATION,
    n_mfcc=N_MFCC,
    n_fft=N_FFT,
    sample_rate=SAMPLE_RATE,
    hop_length=HOP_LENGTH,
    class_path=CLASS_PATH,
    output_path=OUTPUT_PATH,
    env_audio_path=ENV_AUDIO_PATH,
    ignore=True,
    force=False,
):

    audio_path = os.path.join(env_audio_path, f"{str(env_n)}_AUDIO.wav")
    json_path = os.path.join(
        output_path, f"{str(env_n)}_ENV_{n_mfcc}_mfcc_{duration}.json"
    )
    class_map = pd.read_csv(class_path)

    if not os.path.exists(audio_path):
        print("Environment audio not found")
        return

    if os.path.exists(json_path) and not force:
        print(
            f"JSON feature descriptor file already exists for environment #{env_n} - not generating features."
        )
        return

    data = {"mfcc": [], "class_label": []}

    signal, sr = librosa.load(audio_path)

    if sr == sample_rate:
        clip_segments = len(signal) // SAMPLES_PER_SEGMENT

        for s in tqdm(range(clip_segments)):
            start = SAMPLES_PER_SEGMENT * s
            end = start + SAMPLES_PER_SEGMENT

            mfcc = librosa.feature.mfcc(
                y=signal[start:end],
                sr=sample_rate,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            mfcc = mfcc.T
            n_str = str(env_n - 1)
            class_label = class_map[n_str].iloc[s]

            if ignore:
                # logic to skip "edge" cases
                if class_label == "ignore":
                    pass
                elif len(mfcc) == EXPECTED_MFCC_VECTORS_PER_SEGMENT:
                    data["mfcc"].append(mfcc.tolist())
                    data["class_label"].append(str(class_label))
                else:
                    print(f"MFCC vector to segment mismatch at pos {s}")

            else:
                if len(mfcc) == EXPECTED_MFCC_VECTORS_PER_SEGMENT:
                    if class_label == "ignore":
                        class_label = "0"

                    data["mfcc"].append(mfcc.tolist())
                    data["class_label"].append(str(class_label))
                else:
                    print(f"MFCC vector to segment mismatch at pos {s}")

        # save MFCCs to json file
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)


def extract_all_env_features(output_path=OUTPUT_PATH, ignore=True, force=False):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for n in range(1, 7):
        save_env_mfccs(env_n=n, output_path=output_path, ignore=ignore, force=force)

    print(f"MFCCs extracted.")


@click.command()
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force re-generation of features even if features JSON descriptor file already exists.",
)
def extract_env_mfccs_entrypoint(force):
    """Simple program that greets NAME for a total of COUNT times."""
    click.echo("Extracting MFCC features for env set...")
    extract_all_env_features(force=force)


if __name__ == "__main__":
    extract_env_mfccs_entrypoint()
