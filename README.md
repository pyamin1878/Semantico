## Semantico
~A semantic search engine from Wikipedia embeddings~

### Data Collection & Wikipedia API

Through API calls, data was collected through the `wikipediaapi` python package [wikipedia-api](https://pypi.org/project/Wikipedia-API/). 

I chose to pull the `'Featured articles'` for this particular project as they consist of the highest quality data maintained by wiki editors. I provide a simple python script where you can call the Wikipedia API and collect your own data here with some slight modifications: [get_wiki_articles.py](get_wiki_articles.py)

It's important to follow the etiquette guideline when extracting/parsing data from wikipedia. You should take time between requests or implement some form of rate limiting. It is not necessary to web scrape as you can collect quite a large amount of data in a short period through calling the API and Wikipedia prefers you to not web scrape and use the API instead. [API:Main](https://www.mediawiki.org/wiki/API:Main_page)

Refer to the etiquette guide here: [API:Etiquette](https://www.mediawiki.org/wiki/) 

### Data Aggregation & Preprocessing 


