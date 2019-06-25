### Getting started
```
pip install -r requirements.txt
python importer.py --help
```

### Usage
```
usage: importer.py [-h] --input_file str --export_dir str --wav_dir str
                   [--sample_size int] [--random_state int]
                   [--train_size float] [--val_size float]
                   [--noise_ratio float] [--skip_domains bool]

optional arguments:
  -h, --help           show this help message and exit
  --input_file str     path to input file
  --export_dir str     path to export directory
  --wav_dir str        path to wav directory
  --sample_size int    size of sample
  --random_state int   seed for random shuffle
  --train_size float   size of train set
  --val_size float     size of val set
  --noise_ratio float  size of noisy data as a portion of quiet data
  --skip_domains bool  filter out domains
  ```
