## Semantico

| ![alt text](images/logo_.png) |
| :--: |
**Semantic similarity search from word embeddings**


### Data Collection & Wikipedia API

Through API calls, data was collected through the `wikipediaapi` python package [wikipedia-api](https://pypi.org/project/Wikipedia-API/). 

I chose to pull the `'Featured articles'` for this particular project as they consist of the highest quality data maintained by wiki editors. I provide a simple python script where you can call the Wikipedia API and collect your own data here with some slight modifications: [get_wiki_articles.py](get_wiki_articles.py)

It's important to follow the etiquette guideline when extracting/parsing data from Wikipedia. You should take time between requests or implement some form of rate limiting. It is not necessary to web scrape as you can collect quite a large amount of data in a short period through calling the API and Wikipedia prefers you to not web scrape and use the API instead. [API:Main](https://www.mediawiki.org/wiki/API:Main_page)

Refer to the etiquette guide here: [API:Etiquette](https://www.mediawiki.org/wiki/) 

### Dataset

A zip file of the full dataset is provided in the repository in `json` format. Just clone down the repository and unzip `wiki_corpus.zip` in the `/data` dir [wiki_corpus](https://github.com/pyamin1878/Semantico/blob/main/data/wiki_corpus.zip)


### Data Cleaning

**Lowercasing Text**: Standardizing the case of the text to lowercase.

**Removing Special Characters and Punctuation**: Using `regex` to retain only alphanumeric characters and spaces, thus removing any form of punctuation or special characters.

**Replacing Line Breaks**: Converting newline characters `\n` into spaces to maintain sentence continuity.

This process was encapsulated within a `clean_text` function, which was then applied to the dataset to produce a cleaned version.

### Data Preprocessing Steps

**Tokenization**: Splitting the cleaned text into individual words or tokens using NLTK's `word_tokenize`.

**Stop Word Removal**: Eliminating common words that add little value to the analysis, such as "the", "is", etc., using NLTK's predefined list of stop words.

**Lemmatization**: Converting words to their base form, thus reducing the complexity of the vocabulary and consolidating similar forms of a word (e.g., "running" to "run"), using NLTK's `WordNetLemmatizer`.

### SentencePiece 
|*"SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation systems where the vocabulary size is predetermined prior to the neural model training."*|
|:--:|



`SentencePiece` utilizes byte-pair encoding or `BPE` and in my provided [notebook](https://github.com/pyamin1878/Semantico/blob/main/notebooks/data_cleaning_preprocessing.ipynb), SentencePiece was employed to tokenize the preprocessed text data, showcasing its efficacy in creating a manageable and effective representation of text for machine learning models.

The `SentencePieceTrainer.train()` function was used with specified parameters such as input file, model prefix, and vocabulary size. This step is crucial as it adapts the model to the dataset's specific linguistic characteristics.

The `SentencePiece` model was then utilized to tokenize text into subwords or symbols, breaking down complex words into more manageable, model-friendly units. 

For a deeper dive into `BPE` and lossless tokenization refer to the blog post here: 

[Sentencepiece: A simple and language-independent subword tokenizer and detokenizer for neural text processing](https://medium.com/codex/sentencepiece-a-simple-and-language-independent-subword-tokenizer-and-detokenizer-for-neural-text-ffda431e704e)

