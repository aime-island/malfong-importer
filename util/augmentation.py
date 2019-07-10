import soundfile as sf
import librosa as lb
import pandas as pd
import random
import os
from util.modules import sample
import time
import enlighten

def manipulate(file, shift):
    # Breyta pitch
    y, sr = lb.load(file, sr=16000)

    pitch_shifted = lb.effects.pitch_shift(y, sr, shift)

    return pitch_shifted, sr

def augment_data(args, data):
    # Augmenta data

    augmented_data = pd.DataFrame(columns=['wav_filename', 'wav_filesize', 'transcript', 'duration'])
    data_augment = sample(args, data, True)

    random.seed(args.augment_seed)

    shifts = [random.randint(1, 10) for _ in range (len(data_augment))]
    
    pbar = enlighten.Counter(total=len(data_augment), desc='Augmenting train set')

    for i, row in data_augment.iterrows():
        pbar.update()
        # Augmenta þetta sýni og vista undir augmented
        y, sr = manipulate(row['wav_filename'], shifts[i])
        _, filename = os.path.split(row['wav_filename'])
        filepath = os.path.join(args.malromur_dir, 'augmented', filename)
        sf.write(filepath, y, sr, 'PCM_16')
        
        # Búa til new row
        new_row = row
        new_row['wav_filename'] = filepath
        new_row['wav_filesize'] = os.path.getsize(filepath)

        # Bæta new row við augmented_data
        augmented_data = augmented_data.append(row, ignore_index=True)

    # Sameina við data
    new_data = data.append(augmented_data, ignore_index=True)

    return new_data, augmented_data['duration'].sum()