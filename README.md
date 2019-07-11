## Getting started
On Linux, you need to install libsndfile using your distribution’s package manager:
```
sudo apt-get install libsndfile1
```
Quick start:
```
pip install -r requirements.txt

python importer.py --help
```

### Usage
```
usage: importer.py [-h] --malromur_dir str --export_dir str [--duration int]
                   [--max_duration float] [--random_state int]
                   [--train_size float] [--val_size float] [--skip_noise bool]
                   [--skip_domains bool] [--save_corpus bool]
                   [--skip_duplicates bool] [--skip_single_words bool]
                   [--augment float] [--augment_seed int]

Málrómur Importer for DeepSpeech

optional arguments:
  -h, --help                show this help message and exit
  --malromur_dir str        path to malromur
  --export_dir str          path to export directory
  --duration int            size of dataset in seconds
  --max_duration float      max sample duration
  --random_state int        seed for random shuffle
  --train_size float        size of train set
  --val_size float          size of val set
  --skip_noise bool         filter out noisy data
  --skip_domains bool       filter out domains
  --save_corpus bool        save text corpus
  --skip_duplicates bool    use each sentence only once
  --skip_single_words bool  skip single word sentences
  --augment float           size of train set to augment
  --augment_seed int        random state augmentation
  ```
