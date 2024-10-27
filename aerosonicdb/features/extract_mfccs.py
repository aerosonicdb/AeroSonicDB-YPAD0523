#!/usr/bin/env python
"""Feature extraction logic and entrypoint. Calculates MFCC features for all of the audio files and saves it to local file for later use."""
import json
import math
import os

import click
import librosa
import numpy as np
import pandas as pd
from librosa.util import fix_length
from tqdm.auto import tqdm

from aerosonicdb.utils import get_project_root

ROOT_PATH = get_project_root()

# set the default i/o paths
DATASET_PATH = os.path.join(ROOT_PATH, "data/raw")
OUTPUT_PATH = os.path.join(ROOT_PATH, "data/processed")

# set the constants and derivatives
SAMPLE_RATE = 22050
DURATION = 5
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

SAMPLES_PER_SEGMENT = SAMPLE_RATE * DURATION
EXPECTED_MFCC_VECTORS_PER_SEGMENT = math.ceil(SAMPLES_PER_SEGMENT / HOP_LENGTH)


# function to return the audio path for a given file
def get_audio_path(df, filename, dataset_path=DATASET_PATH):
    sep = os.sep
    audio_dir = dataset_path + sep + "audio" + sep
    audio_class = df.loc[df["filename"] == filename, "class"].values[0]
    path = audio_dir + audio_class.astype(str) + sep + filename
    return path


def save_mfccs(
    dataset_path=DATASET_PATH,
    output_path=OUTPUT_PATH,
    n_mfcc=N_MFCC,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    duration=DURATION,
    sample_rate=SAMPLE_RATE,
    set_str="test",
    force=False,
):
    """
    function to load audio, split into 5 second segments, transform to MFCCs then save to a JSON file.

    """
    meta_path = os.path.join(dataset_path, "sample_meta.csv")

    # check i/o paths exist - warn or create if not
    if not os.path.exists(dataset_path):
        print(
            f"Dataset path not found. Call aerosonicdb.data.download() to download to the default path. "
            f'Alternatively, add custom path to dataset location with the parameter "dataset_path".'
        )
        return

    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print(f"Output path created. Features will be saved to: {output_path}")

    json_path = os.path.join(output_path, f"{n_mfcc}_mfcc_{duration}_{set_str}.json")

    if os.path.exists(json_path) and not force:
        print(f"{set_str} MFCCs extracted.")
        return

    # create a dictionary to store the JSON data

    data = {"mfcc": [], "class_label": [], "subclass_label": [], "fold_label": []}
    df = pd.read_csv(meta_path)
    df = df[df["train-test"] == set_str].reset_index(drop=True)

    for row in tqdm(df.index):
        base_filename = df["filename"].iloc[row]

        audio_path = get_audio_path(
            df=df, filename=base_filename, dataset_path=dataset_path
        )

        class_label = df["class"].iloc[row]
        subclass_label = df["subclass"].iloc[row]
        fold_label = df["fold"].iloc[row]
        offset = df["offset"].iloc[row]
        clip_duration = df["duration"].iloc[row]

        signal, sr = librosa.load(audio_path, offset=offset, duration=clip_duration)

        if sr == sample_rate:
            clip_segments = int(np.ceil(len(signal) / SAMPLES_PER_SEGMENT))

            for s in range(clip_segments):
                start = SAMPLES_PER_SEGMENT * s
                end = start + SAMPLES_PER_SEGMENT

                if len(signal[start:]) < SAMPLES_PER_SEGMENT:
                    stub = signal[start:]
                    padded = fix_length(stub, size=int(5 * sr))
                    mfcc = librosa.feature.mfcc(
                        y=padded,
                        sr=sample_rate,
                        n_mfcc=n_mfcc,
                        n_fft=n_fft,
                        hop_length=hop_length,
                    )
                else:
                    mfcc = librosa.feature.mfcc(
                        y=signal[start:end],
                        sr=sample_rate,
                        n_mfcc=n_mfcc,
                        n_fft=n_fft,
                        hop_length=hop_length,
                    )

                mfcc = mfcc.T

                if len(mfcc) == EXPECTED_MFCC_VECTORS_PER_SEGMENT:
                    data["mfcc"].append(mfcc.tolist())
                    data["class_label"].append(str(class_label))
                    data["subclass_label"].append(str(subclass_label))
                    data["fold_label"].append(str(fold_label))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


@click.command()
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force re-generation of features even if features JSON descriptor file already exists.",
)
def extract_mfccs_entrypoint(force):
    """Simple program that greets NAME for a total of COUNT times."""
    click.echo("Extracting MFCC features for train set...")
    save_mfccs(
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_PATH,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        set_str="train",
        force=force,
    )

    click.echo("Extracting MFCC features for test set...")
    save_mfccs(
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_PATH,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        set_str="test",
        force=force,
    )


if __name__ == "__main__":
    extract_mfccs_entrypoint()
