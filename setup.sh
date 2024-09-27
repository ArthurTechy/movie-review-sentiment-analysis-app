#!/bin/bash

# Download NLTK data
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('vader_lexicon')"