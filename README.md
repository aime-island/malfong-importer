### Getting started
```
pip install -r requirements.txt
python importer.py --help
```

### Usage
```
usage: importer.py [-h] --input_file str --export_dir str --wav_dir str
                   [--sample_size int] [--duration float]
                   [--max_duration float] [--random_state int]
                   [--train_size float] [--val_size float] [--skip_noise bool]
                   [--skip_domains bool] [--save_corpus bool]

optional arguments:
  -h, --help            show this help message and exit
  --input_file str      path to input file
  --export_dir str      path to export directory
  --wav_dir str         path to wav directory
  --sample_size int     size of sample
  --duration float      size of dataset in seconds
  --max_duration float  max sample duration
  --random_state int    seed for random shuffle
  --train_size float    size of train set
  --val_size float      size of val set
  --skip_noise bool     filter out noisy data
  --skip_domains bool   filter out domains
  --save_corpus bool    save text corpus
  ```
