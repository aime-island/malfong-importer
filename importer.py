from util.my_argparser import create_parser
from util.modules import read_data, filter_data, split_data, export_csv
import os

parser = create_parser()

def main():
    # Lesa inn data
    data = read_data(args)

    # Filtera data
    data = filter_data(args, data)
    
    # Ef corpus er umbeðið
    if (args.save_corpus):
        data['transcript'].to_csv(os.path.join(args.export_dir, 'text.txt'), header=False, index=False, mode='a', encoding='utf-8')
    
    # Splitta data
    train, val, test = split_data(args, data)
    print('Train duration', int(train['duration'].sum()))
    print('Val duration', int(val['duration'].sum()))
    print('Test duration', int(test['duration'].sum()))
    print('...')

    # Exporta data
    export_csv(args, train, 'train')
    export_csv(args, val, 'val')
    export_csv(args, test, 'test')
    print("Task done.")
    print("")
    print("Three files created at: ", args.export_dir)

if __name__ == "__main__":
    args = parser.parse_args()
    main()