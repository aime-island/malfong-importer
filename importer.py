from util.argparser import create_parser
from util.modules import (read_data, filter_data, sample,
    split_data, export_csv, export_corpus, make_dirs, format_data)
from util.augmentation import augment_data

parser = create_parser()

def main():
    # Lesa inn data
    data = read_data(args)

    # Filtera data
    data = filter_data(args, data)
    
    # Vista texta málheild
    if (args.save_corpus):
        export_corpus(args, data)

    # Random shuffla og stilla stærð
    data = sample(args, data)

    # Formatta data
    data = format_data(args, data)

    # Splitta data
    train, val, test = split_data(args, data)

    # Búa til directories
    make_dirs(args)

    # Augmenta data
    aug_duration = 0.0
    if (args.augment):
        train, aug_duration = augment_data(args, train)

    print('\n')
    print('Train duration: %s seconds, including %s augmented' 
        %(int(train['duration'].sum()), int(aug_duration)))
    print('Val duration: %s seconds, max value: %s seconds' 
        %(int(val['duration'].sum()), val['duration'].max()))
    print('Test duration: %s seconds, max value: %s seconds' 
        %(int(test['duration'].sum()), test['duration'].max()))

    # Exporta data
    export_csv(args, train, 'train')
    export_csv(args, val, 'val')
    export_csv(args, test, 'test')

    # Log results
    print('Task done. \n')

    numfiles = 'Three'
    if (args.save_corpus):
        numfiles = 'Four'
    print('%s files created at: %s' %(numfiles, args.export_dir))

if __name__ == "__main__":
    args = parser.parse_args()
    main()