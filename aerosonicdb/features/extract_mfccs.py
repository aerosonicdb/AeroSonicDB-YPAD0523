import os
import math
import json
import pandas as pd
import numpy as np
import librosa
from aerosonicdb.utils import get_project_root


root_path = get_project_root()

# set the default i/o paths
dataset_path = os.path.join(root_path, 'data/raw')
output_path = os.path.join(root_path, 'data/processed')

# set the constants and derivatives
sample_rate = 22050
duration = 5
n_mfcc = 13
n_fft = 2048
hop_length = 512

samples_per_segment = sample_rate * duration
expected_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)


# function to return the audio path for a given file
def get_audio_path(df, filename, dataset_path=dataset_path):
    sep = os.sep
    audio_dir = dataset_path + sep + 'audio' + sep
    audio_class = df.loc[df['filename'] == filename, 'class'].values[0]
    path = audio_dir + audio_class.astype(str) + sep + filename
    return path


# function to load audio, split into 5 second segments, transform to MFCCs then save to a JSON file
def save_mfccs(dataset_path=dataset_path,
               output_path=output_path,
               n_mfcc=n_mfcc,
               n_fft=n_fft,
               hop_length=hop_length,
               set_str='test'):

    meta_path = os.path.join(dataset_path, 'sample_meta.csv')

    # check i/o paths exist - warn or create if not
    if not os.path.exists(dataset_path):
        print(f'Dataset path not found. Call aerosonicdb.data.download() to download to the default path. '
              f'Alternatively, add custom path to dataset location with the parameter "dataset_path".')

    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print(f'Output path created. Features will be saved to: {output_path}')

    json_path = os.path.join(output_path, f'{n_mfcc}_mfcc_{duration}_{set_str}.json')
    
    if not os.path.exists(json_path):

        # create a dictionary to store the JSON data

        data = {
            'mfcc': [],
            'class_label': [],
            'subclass_label': [],
            'fold_label': []
        }

        df = pd.read_csv(meta_path)

        df = df[df['train-test'] == set_str].reset_index(drop=True)

        for row in df.index:
            base_filename = df['filename'].iloc[row]

            audio_path = get_audio_path(df=df, filename=base_filename, dataset_path=dataset_path)

            class_label = df['class'].iloc[row]
            subclass_label = df['subclass'].iloc[row]
            fold_label = df['fold'].iloc[row]

            signal, sr = librosa.load(audio_path)

            if sr == sample_rate:

                # data['filename'].append(base_filename)

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

                    if len(mfcc) == expected_mfcc_vectors_per_segment:
                        data['mfcc'].append(mfcc.tolist())
                        data['class_label'].append(str(class_label))
                        data['subclass_label'].append(str(subclass_label))
                        data['fold_label'].append(str(fold_label))

        # save MFCCs to json file
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)
    else:
        print(f'{set_str} features already extracted.')
        

if __name__ == "__main__":
    save_mfccs(dataset_path=dataset_path,
               output_path=output_path,
               n_mfcc=n_mfcc,
               n_fft=n_fft,
               hop_length=hop_length,
               set_str='train')

    save_mfccs(dataset_path=dataset_path,
               output_path=output_path,
               n_mfcc=n_mfcc,
               n_fft=n_fft,
               hop_length=hop_length,
               set_str='test')