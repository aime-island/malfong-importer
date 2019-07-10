import sys, os
import numpy as np
import pandas as pd
import platform
from sklearn.model_selection import train_test_split

# Nöfn dálka í dataframe
column_names=['name', 'environment', 
        'number', 'gender', 
        'age', 'transcript', 
        'duration', 'classification']

def read_data(args):
    # Lesa inn efni
    data = pd.read_csv(os.path.join(args.malromur_dir, 'info.txt'), names=column_names)

    # Nota bara efni úr correct
    data = data.loc[data['classification'] == 'correct']
    
    # Breyta quite vs noise í flokkabreytu
    data['environment'] = data['environment'].apply(lambda x: '1' if 'Quiet' in x else 0)
    data['environment'] = data['environment'].astype('int64')

    return data
    
def filter_data(args, data):
    # Filtera út illegal characters
    if (args.skip_domains):
        searchfor = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', r'\.', u'\ufeff', u'\u003B']
        filters = data[data['transcript'].str.contains('|'.join(searchfor))].index
        data = data.drop(filters)

    # Filtera út noisy data
    if (args.skip_noise):
        data = data.loc[data['environment'] == 1]
    
    # Filtera út löng hljóðbrot
    data = data.loc[data['duration'] < args.max_duration]

    # Filtera út duplicates, eyðir öllu nema fyrsta occurance út
    if (args.skip_duplicates):
        data = data.drop_duplicates(subset='transcript')

    # Filtera út setningar með bara eitt orð
    if (args.skip_single_words):
        data = data[data['transcript'].str.contains(' ')]
        
    return data

def sample(args, data, augmentation=False):
    # Stilla stærð gagnasetts
    duration = args.duration
    random_state = args.random_state

    if (augmentation):
        duration = args.augment * data['duration'].sum()
        random_state = args.augment_seed

    # Shuffla gögnunum með random state og reseta indexes
    data = data.sample(frac=1, random_state=random_state)
    data = data.reset_index(drop=True)
    
    # Stilla stærð gagnasetts í sekúndum
    if (data['duration'].sum() > duration):
        sum = 0
        for i, row in data.iterrows():
            sum += row['duration']
            if (sum > duration):
                data = data.head(i+1)
                break

    return data

def split_data(args, data):
    # Splita í train, val, test

    train_seconds = data['duration'].sum() * args.train_size
    val_seconds = data['duration'].sum() * args.val_size
        
    sum = 0
    for i, row in data.iterrows():
        sum += row['duration']
        if (sum > train_seconds):
            train = data.head(i+1)
            data.drop(train.index, inplace=True)

            # Reset index fyrir rest
            data = data.reset_index(drop=True)
            sum = 0
            for j, row in data.iterrows():
                sum += row['duration']
                if (sum > val_seconds):
                    val = data.head(j+1)
                    data.drop(data.head(j+1).index, inplace=True)
                    test = data
                    return train, val, test

def make_dirs(args):
    # Búa til directories ef þau eru ekki til
    if not os.path.exists(args.export_dir):
        os.makedirs(args.export_dir)
    if (args.augment):
        augmented = os.path.join(args.malromur_dir, 'augmented')
        if not os.path.exists(augmented):
            os.makedirs(augmented)

def format_data(args, data):

    # Búa til nýtt dataframe með þeim dálkum sem við þurfum
    filesizes = np.zeros(len(data))
    new_data = pd.DataFrame(
        pd.np.column_stack([data['name'], filesizes, data['transcript'].str.lower(), data['duration']]), 
        columns=['wav_filename', 'wav_filesize', 'transcript', 'duration']
        )

    # Laga path á audio files og reikna filesize
    for _, row in new_data.iterrows():
        # Ef ekki linux, þá laga filename
        if (platform.system() != 'Linux'):
            row['wav_filename'] = row['wav_filename'].replace(":", "_")
        row['wav_filename'] = os.path.join(args.malromur_dir, 'correct', row['wav_filename'] + '.wav')
        row['wav_filesize'] = os.path.getsize(row['wav_filename'])

    return new_data

def export_corpus(args, data):
    transcript = data['transcript'].str.lower()
    transcript.to_csv(os.path.join(args.export_dir, 'text.txt'), header=False, index=False, encoding='utf-8')
    print('Text corpus generated with %s lines' %(len(transcript)))
    print('\n')

def export_csv(args, data, name):
    # Exporta í csv
    data.to_csv(os.path.join(args.export_dir, name + '.csv'), encoding='utf-8', index=None, header=True)