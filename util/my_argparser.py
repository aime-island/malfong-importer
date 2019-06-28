import argparse

def create_parser():
    parser = argparse.ArgumentParser(
        description='Importer for DeepSpeech',
        add_help=True,
        formatter_class=argparse.MetavarTypeHelpFormatter)

    parser.add_argument(
        '--input_file', required=True, type=str, help='path to input file')
    parser.add_argument(
        '--export_dir', required=True, type=str, help='path to export directory')
    parser.add_argument(
        '--wav_dir', required=True, type=str, help='path to wav directory')
    parser.add_argument(
        '--sample_size', required=False, type=int, help='size of sample')
    parser.add_argument(
        '--duration_hours', required=False, type=float, help='size of dataset in hours')
    parser.add_argument(
        '--random_state', required=False, type=int, help='seed for random shuffle')
    parser.add_argument(
        '--train_size', required=False, type=float, help='size of train set', default=0.7)
    parser.add_argument(
        '--val_size', required=False, type=float, help='size of val set', default=0.2)
    parser.add_argument(
        '--skip_noise', required=False, type=bool, help='filter out noisy data', default=False)
    parser.add_argument(
        '--skip_domains', required=False, type=bool, help='filter out domains')
    
    return parser