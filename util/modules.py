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
    data = pd.read_csv(args.input_file, names=column_names)

    # Nota bara efni úr correct
    data = data.loc[data['classification'] == 'correct']
    
    # Breyta quite vs noise í flokkabreytu
    data['environment'] = data['environment'].apply(lambda x: '1' if 'Quiet' in x else 0)
    data['environment'] = data['environment'].astype('int64')

    return data
    
def filter_data(args, data):

    # Sleppa setningum með filters
    if (args.skip_domains):
        filters = data[data['transcript'].str.contains(r'\.')].index
        searchfor = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        filters2 = data[data['transcript'].str.contains('|'.join(searchfor))].index
        data = data.drop(filters)
        data = data.drop(filters2)
        filters3 = data[data['transcript'].str.contains(u'\ufeff')].index
        data = data.drop(filters3)
        filters4 = data[data['transcript'].str.contains(u'\u003B')].index
        data = data.drop(filters4)
    
    # Filtera út noisy data
    if (args.skip_noise):
        data = data.loc[data['environment'] == 1]
    
    return data

def split_data(args, data):
    # Splita í train, val, test, halda hlutföllum af environment í settunum

    # Stilla stærð gagnasetts
    if (args.sample_size and len(data) > args.sample_size):
        data = data.sample(n=args.sample_size, random_state=args.random_state)
    
    # Stilla stærð gagnasetts miðað við tíma
    elif (args.duration and data['duration'].sum() > args.duration):

        # Shuffla gögnunum með random state og reseta indexes
        data = data.sample(frac=1, random_state=args.random_state)
        data = data.reset_index(drop=True)
        sum = 0
        for i, row in data.iterrows():
            sum += row['duration']
            if (sum > args.duration):
                data = data.head(i+1)
                break

    # Ef það er beðið um eina setningu skila eins í train, val og test
    if (len(data) == 1):
        return data, data, data

    # Ef það var beðið um tímaskiptingu
    elif (args.duration):
        train_seconds = args.duration * args.train_size
        val_seconds = train_seconds * args.val_size
        sum = 0
        for i, row in data.iterrows():
            sum += row['duration']
            if (sum > train_seconds):
                train = data.head(i+1)
                data.drop(data.head(i+1).index, inplace=True)
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
    
    # Ef það er ekki beðið um tímaskiptingu
    else:
        main, test = train_test_split(data, test_size=1-args.train_size, random_state=args.random_state)
        train, val = train_test_split(main, test_size=args.val_size, random_state=args.random_state)
        return train, val, test

def format_data(args, data):

    # Búa til nýtt dataframe með þeim dálkum sem við þurfum
    filesizes = np.zeros(len(data))
    new_data = pd.DataFrame(
        pd.np.column_stack([data['name'], filesizes, data['transcript'].str.lower()]), 
        columns=['wav_filename', 'wav_filesize', 'transcript']
        )

    # Laga path á audio files og reikna filesize
    for _, row in new_data.iterrows():
        # Ef ekki linux, þá laga filename
        if (platform.system() != 'Linux'):
            row['wav_filename'] = row['wav_filename'].replace(":", "_")
        row['wav_filename'] = os.path.join(args.wav_dir, row['wav_filename'] + '.wav')
        row['wav_filesize'] = os.path.getsize(row['wav_filename'])

    return new_data

def export_corpus(args, data):
    transcript = data['transcript'].str.lower()
    transcript.to_csv(os.path.join(args.export_dir, 'text.txt'), header=False, index=False, encoding='utf-8')
    print('Text corpus generated with %s lines' %(len(transcript)))
    print('')

def export_csv(args, data, name):
    # Exporta í csv
    data = format_data(args, data)
    data.to_csv(os.path.join(args.export_dir, name + '.csv'), encoding='utf-8', index=None, header=True)