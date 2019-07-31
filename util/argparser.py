import argparse

def create_parser():
    parser = argparse.ArgumentParser(
        description='Importer for DeepSpeech',
        add_help=True,
        formatter_class=argparse.MetavarTypeHelpFormatter)

    parser.add_argument(
        '--malromur_dir', required=True, type=str, help='path to malromur')
    parser.add_argument(
        '--export_dir', required=True, type=str, help='path to export directory')
    parser.add_argument(
        '--duration', required=False, type=int, help='size of dataset in seconds')
    parser.add_argument(
        '--max_duration', required=False, type=float, help='max sample duration')
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
    parser.add_argument(
        '--save_corpus', required=False, type=bool, help='save text corpus', default=False)
    parser.add_argument(
        '--skip_duplicates', required=False, type=bool, help='use each sentence only once', default=False)
    parser.add_argument(
        '--skip_single_words', required=False, type=bool, help='skip single word sentences', default=False)
    parser.add_argument(
        '--augment', required=False, type=float, help='size of train set to augment')
    parser.add_argument(
        '--augment_seed', required=False, type=int, help='random state for what to augment')
    parser.add_argument(
        '--ru_split', required=False, type=bool, help='train test splits like tala.ru.is', default=False)
    parser.add_argument(
        '--convert_domains', required=False, type=bool, help='convert domain names to name punktur domain', default=False)
    parser.add_argument(
        '--only_correct', required=False, type=bool, help='only use utterances from correct folder', default=False)
    return parser