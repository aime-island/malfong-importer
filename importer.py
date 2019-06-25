from util.my_argparser import create_parser
from util.modules import read_data, filter_data, split_data, export_csv

parser = create_parser()

def main():
    data = read_data(args)
    data = filter_data(args, data)
    train, val, test = split_data(args, data)
    export_csv(args, train, 'train')
    export_csv(args, val, 'val')
    export_csv(args, test, 'test')
    print("Task done.")
    print("")
    print("Three files created at: ", args.wav_dir)

if __name__ == "__main__":
    args = parser.parse_args()
    main()