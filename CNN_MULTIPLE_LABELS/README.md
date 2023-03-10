To run
----
python CNNJobMatching_2Label.py

Requirements
----
- Python 3.6
- gensim
- xlrd
- spacy
- numpy
- xlsxwriter
- keras
- sklearn

Installation
----
To install the necessary libraries you could use pip:

pip install gensim xlrd spacy numpy xlsxwriter keras sklearn

Download:

Pre-trained embeddings used wiki-news-300d-1M.vec for word2vec. Download from https://www.kaggle.com/yesbutwhatdoesitmean/wikinews300d1mvec
Place it in the CNNJobMatching_3Label directory

Run in your terminal for the environment:

python -m nltk.downloader all
python -m spacy download en_core_web_lg

Configuration
----
It's essential to change the following variables at the top of the script:

input file -> Excel file for training, first column should contain descriptions, second column labels, and third column
sub labels

Output File
----

The output file will be an Excel file with the first column being the descriptions given, the second and third column being the human
label and sub label respectively. The fourth and fifth column will be the AI label and sub label respectively. There will
also be a metrics chart
