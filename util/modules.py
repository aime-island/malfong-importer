import sys, os
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
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
    # Sleppa setningum með . í (lénum)
    if (args.skip_domains):
        filters = data[data['transcript'].str.contains(r'\.')].index
        data = data.drop(filters)

    # Stilla ratio af noisy audio
    if (args.noise_ratio):
        # Beyta random under sampler á values
        rus = RandomUnderSampler(sampling_strategy=args.noise_ratio, random_state=args.random_state)
        data_res, _ = rus.fit_resample(data.values, data['environment'].values)

        # Breyta gögnum aftur í dataframe
        data = pd.DataFrame(data_res, columns=column_names)
    
    # Stilla stærð gagnasetts í endann
    if (args.sample_size and len(data) > args.sample_size):
        data = data.sample(n=args.sample_size, random_state=args.random_state)

    return data

def split_data(args, data):
    # Splita í train, val, test, halda hlutföllum af environment í settunum
    if (len(data) < 100):
        return data, data, data
    
    main, test = train_test_split(data, test_size=args.train_size, random_state=args.random_state, stratify=data['environment'])
    train, val = train_test_split(main, test_size=args.val_size, random_state=args.random_state, stratify=main['environment'])

    return train, val, test

def format_data(args, data):
    # Búa til nýtt dataframe með þeim dálkum sem við þurfum
    filesizes = np.zeros(len(data))
    new_data = pd.DataFrame(
        pd.np.column_stack([data['name'], filesizes, data['transcript']]), 
        columns=['wav_filename', 'wav_filesize', 'transcript']
        )

    # Laga path á audio files og reikna filesize
    for _, row in new_data.iterrows():
        row['wav_filename'] = row['wav_filename'].replace(":", "_")
        row['wav_filename'] = args.wav_dir + '/' + row['wav_filename'] + '.wav'
        row['wav_filesize'] = os.path.getsize(row['wav_filename'])
    
    return new_data

def export_csv(args, data, name):
    # Exporta í csv
    data = format_data(args, data)
    data.to_csv(args.export_dir + '/' + name + '.csv', encoding='utf-8', index=None, header=True)