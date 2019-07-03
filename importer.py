from util.my_argparser import create_parser
from util.modules import read_data, filter_data, split_data, export_csv, export_corpus

parser = create_parser()

def main():
    # Lesa inn data
    data = read_data(args)

    # Filtera data
    data = filter_data(args, data)
    
    # Ef corpus er umbeðið
    if (args.save_corpus):
        export_corpus(args, data)
    
    # Splitta data
    train, val, test = split_data(args, data)
    print('Train duration: %s seconds, max value: %s seconds' 
        %(int(train['duration'].sum()), train['duration'].max()))
    print('Val duration: %s seconds, max value: %s seconds' 
        %(int(val['duration'].sum()), val['duration'].max()))
    print('Test duration: %s seconds, max value: %s seconds' 
        %(int(test['duration'].sum()), test['duration'].max()))
    print('\n Calculating... \n')

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