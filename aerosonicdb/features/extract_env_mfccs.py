import os
import math
import json
import pandas as pd
import numpy as np
import librosa
from aerosonicdb.utils import get_project_root

root_path = get_project_root()

# set the i/o paths
dataset_path = os.path.join(root_path, 'data/raw')
env_audio_path = os.path.join(dataset_path, 'env_audio')
output_path = os.path.join(root_path, 'data/processed')
class_path = os.path.join(dataset_path, 'environment_class_mappings.csv')

# set the constants and derivatives
sample_rate = 22050
duration = 5
n_mfcc = 13
n_fft = 2048
hop_length = 512

samples_per_segment = sample_rate * duration
expected_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)


def save_env_mfccs(env_n,
                   duration=duration,
                   n_mfcc=n_mfcc,
                   n_fft=n_fft,
                   hop_length=hop_length,
                   class_path=class_path,
                   output_path=output_path):

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

            clip_segments = len(signal) // samples_per_segment

            for s in range(clip_segments):
                start = samples_per_segment * s
                end = start + samples_per_segment

                mfcc = librosa.feature.mfcc(y=signal[start:end],
                                            sr=sample_rate,
                                            n_mfcc=n_mfcc,
                                            n_fft=n_fft,
                                            hop_length=hop_length)
                mfcc = mfcc.T

                n_str = str(env_n - 1)

                class_label = class_map[n_str].iloc[s]

                if len(mfcc) == expected_mfcc_vectors_per_segment:
                    data['mfcc'].append(mfcc.tolist())
                    data['class_label'].append(str(class_label))

        # save MFCCs to json file
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)

    else:
        print(f'MFCCs already extracted. See {output_path}')
        

if __name__ == "__main__":
    for i in range(0, 6):
        n = i + 1
        save_env_mfccs(
            env_n=n,
            output_path=output_path,
            duration=duration,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            class_path=class_path)
