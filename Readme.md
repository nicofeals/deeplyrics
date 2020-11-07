# DeepLyrics: RNN powered lyric generator

DeepLyrics is an Recurrent Neural Network powered lyrics generator, based on the most popular metalcore bands' most popular song lyrics.

## Architecture
It's quite a basic model, that consists of one Embedding layer, two stacked GRU layers, and one Dense layer.

## Dataset
The dataset was built using scraper.py and merger.py. The objective is to collect as many songs lyrics as wanted and required, relevant ones, and merge them in a file
### Scraper
`scraper.py` uses the `lyricsgenius` library to collect the lyrics from the Genius API. Arists to collect songs from are listed at the beginning of the file.  
For each artist, we collect their 20 most popular songs, and then save the results in a json file.  
To run it, you must
- create a Genius API account
- generate an access token
```shell
pip install lyricsgenius
export GENIUS_ACCESS_TOKEN=<your_access_token>
python scraper.py
```
### Merger
`merger.py` will look for all json files in a given directory (the directory where the lyrics got saved by the scraper), and extract the `lyrics` field from each file and append its value to a file (here `merged_lyrics_metalcore.txt`).
This gives a file with thousands of line holding all collected lyrics.
To run it, you must
- ensure the filenames and directory names are set accordingly to your situation/context
```shell
python merger.py
```
### Shuffler
`shuffler.py` takes the previously generated text file, and suffles paragraphs (to keep the meaning within paragraphs), and saves the result into a new file.  
This allows for data augmentation and doubles the size of the dataset, which should give better results when training the model  
To run it, you must
- ensure the filenames are set accordingly to your situation/context
```shell
python shuffler.py
```

## Training
The model was trained using the `train.ipynb` notebook  
It builds a model with the given architecture, that has 1024 RNN units for each GRU layer and an embedding dimension of 256.  
Other hyper parameters had to be set, such as the input sequence length. I had to try out multiple values before finding one that suited my needs and generated good results.  
If the sequence is too short (~30 characters), the model will only train on maximum 1.5 sentences, and won't really be aware of paragraphs. If the sequence length is too long (>150 characters), the model will have a tendency to replicate whole sentences, and that's not exactly what we want.  
120 characters seems to be a good compromise, when there's close to no spelling mistakes, and not a lot of original sentences replicated. The perfect (or at least a better) choice has yet to be found.  
The model is trained on 100 epochs, with an early stop callback to avoid overfitting, and the best weights get saved. The loss of this version is about 0.1, which is quite good.  