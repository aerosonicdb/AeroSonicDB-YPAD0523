import os
import math
import json
import pandas as pd
import librosa
from aerosonicdb.utils import get_project_root

ROOT_PATH = get_project_root()

# set the i/o paths
DATASET_PATH = os.path.join(ROOT_PATH, 'data/raw')
ENV_AUDIO_PATH = os.path.join(DATASET_PATH, 'env_audio')
OUTPUT_PATH = os.path.join(ROOT_PATH, 'data/processed')
CLASS_PATH = os.path.join(DATASET_PATH, 'environment_class_mappings.csv')

# set the constants and derivatives
SAMPLE_RATE = 22050
DURATION = 5
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

SAMPLES_PER_SEGMENT = SAMPLE_RATE * DURATION
EXPECTED_MFCC_VECTORS_PER_SEGMENT = math.ceil(SAMPLES_PER_SEGMENT / HOP_LENGTH)


def save_env_mfccs(env_n,
                   duration=DURATION,
                   n_mfcc=N_MFCC,
                   n_fft=N_FFT,
                   sample_rate=SAMPLE_RATE,
                   hop_length=HOP_LENGTH,
                   class_path=CLASS_PATH,
                   output_path=OUTPUT_PATH,
                   env_audio_path=ENV_AUDIO_PATH):
    audio_path = os.path.join(env_audio_path, f'{str(env_n)}_AUDIO.wav')
    json_path = os.path.join(output_path, f'{str(env_n)}_ENV_{n_mfcc}_mfcc_{duration}.json')
    class_map = pd.read_csv(class_path)

    if not os.path.exists(audio_path):
        print('Environment audio not found')

    if not os.path.exists(json_path):

        data = {
            'mfcc': [],
            'class_label': []
        }

        signal, sr = librosa.load(audio_path)

        if sr == sample_rate:

            clip_segments = len(signal) // SAMPLES_PER_SEGMENT

            for s in range(clip_segments):
                start = SAMPLES_PER_SEGMENT * s
                end = start + SAMPLES_PER_SEGMENT

                mfcc = librosa.feature.mfcc(y=signal[start:end],
                                            sr=sample_rate,
                                            n_mfcc=n_mfcc,
                                            n_fft=n_fft,
                                            hop_length=hop_length)
                mfcc = mfcc.T

                n_str = str(env_n - 1)

                class_label = class_map[n_str].iloc[s]

                if len(mfcc) == EXPECTED_MFCC_VECTORS_PER_SEGMENT:
                    data['mfcc'].append(mfcc.tolist())
                    data['class_label'].append(str(class_label))

        # save MFCCs to json file
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)

    else:
        print(f'MFCCs already extracted. See {output_path}')


def extract_all_env_feats():
    for i in range(0, 6):
        n = i + 1
        save_env_mfccs(env_n=n)


if __name__ == "__main__":
    extract_all_env_feats()
