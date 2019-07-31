import sys
from os import makedirs
from os.path import join, getsize, exists
import numpy as np
import pandas as pd
import platform
from sklearn.model_selection import train_test_split
import enlighten

# Nöfn dálka í dataframe
column_names=['name', 'environment', 
        'number', 'gender', 
        'age', 'transcript', 
        'duration', 'classification']

def read_data(args):
    # Lesa inn efni
    data = pd.read_csv(join(args.malromur_dir, 'info.txt'), names=column_names)

    return data

def ru_split(args, data):
    test_all = pd.read_csv('./resources/test_all_lrec2018', delim_whitespace=True, usecols=[0], names=['filename'])
    ru_test = data[data['name'].isin(test_all['filename'])]
    ru_train = data.drop(ru_test.index)
    return ru_train, ru_test

def filter_data(args, data):

    # Nota bara efni úr correct
    if args.only_correct:
        data = data.loc[data['classification'] == 'correct']
    
    if (args.skip_domains):
        searchfor = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', r'\.', u'\ufeff', u'\u003B']
        filters = data[data['transcript'].str.contains('|'.join(searchfor))].index
        data = data.drop(filters)

    # Filtera út noisy data

    # Breyta quite vs noise í flokkabreytu
    data['environment'] = data['environment'].apply(lambda x: '1' if 'Quiet' in x else 0)
    data['environment'] = data['environment'].astype('int64')
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
    if (duration):
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
    if args.ru_split:
        main, test = ru_split(args, data)
        main = filter_data(args, main)
        train_seconds = main['duration'].sum() * args.train_size
        data = main
    else:
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
            if args.ru_split:
                val = data
                return train, val, test
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
    if not exists(args.export_dir):
        makedirs(args.export_dir)
    if (args.augment):
        augmented = join(args.malromur_dir, 'augmented')
        if not exists(augmented):
            makedirs(augmented)

def format_data(args, data):

    # Búa til nýtt dataframe með þeim dálkum sem við þurfum
    filesizes = np.zeros(len(data))
    new_data = pd.DataFrame(
        pd.np.column_stack([data['name'], filesizes, data['transcript'].str.lower(), data['duration'], data['classification']]), 
        columns=['wav_filename', 'wav_filesize', 'transcript', 'duration', 'classification']
        )

    # Laga path á audio files og reikna filesize
    pbar = enlighten.Counter(total=len(data), desc='Formatting data')
    for _, row in new_data.iterrows():
        # Ef ekki linux, þá laga filename
        if (platform.system() != 'Linux'):
            row['wav_filename'] = row['wav_filename'].replace(":", "_")
        filename = row['wav_filename'] + '.wav'
        if (row['classification'] == 'correct'):
            row['wav_filename'] = join(args.malromur_dir, 'correct', filename)
        elif (row['classification'] == 'incorrect'):
            row['wav_filename'] = join(args.malromur_dir, 'incorrect_utts', filename)
        elif (row['classification'] == 'incomplete'):
            row['wav_filename'] = join(args.malromur_dir, 'incomplete_utts', filename)
        elif (row['classification'] == 'unclear'):
            row['wav_filename'] = join(args.malromur_dir, 'unclear_utts', filename)
        else:
            row['wav_filename'] = join(args.malromur_dir, 'additional_speech', filename)

        row['wav_filesize'] = getsize(row['wav_filename'])
        pbar.update()
    return new_data

def export_corpus(args, data):
    transcript = data['transcript'].str.lower()
    transcript.to_csv(join(args.export_dir, 'text.txt'), header=False, index=False, encoding='utf-8')
    print('Text corpus generated with %s lines' %(len(transcript)))
    print('\n')

def export_csv(args, data, name):
    # Exporta í csv
    data.to_csv(join(args.export_dir, name + '.csv'), columns=['wav_filename', 'wav_filesize', 'transcript'], encoding='utf-8', index=None, header=True)