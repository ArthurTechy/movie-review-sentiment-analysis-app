# Movie Review Sentiment Analysis

## Overview
This Streamlit application implements a binary sentiment classification model for movie reviews using the Stanford Large Movie Review Dataset. Users can analyze the sentiment of movie reviews through manual input or by uploading a CSV file.

## Features
- Sentiment prediction for individual movie reviews
- Batch processing of reviews via CSV upload
- Word cloud generation for visual representation of review content
- VADER sentiment analysis for additional insight
- Confidence histogram for uploaded reviews
- Interactive web interface using Streamlit

## Dataset
- **Source**: Stanford Large Movie Review Dataset
- **Size**: 25,000 reviews for training, 25,000 for testing
- **Link**: [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
- **Citation**: Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

## Model
- **Algorithm**: Logistic Regression
- **Accuracy**: 0.8776
- **Mean CV score**: 0.8579

## Installation
```bash
# Clone the repository
git clone https://github.com/ArthurTechy/movie-review-sentiment-analysis-app.git
cd movie-review-sentiment-analysis-app

# Install required packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

## Usage
To run the Streamlit app:
```bash
streamlit run app.py
```

## File Structure
- `app.py`: Main Streamlit application
- `logistic_regression_model.pkl`: Trained Logistic Regression model
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer
- `requirements.txt`: List of required Python packages
- `setup.sh`: Script for setting up the environment (particularly for deployment)

## Dependencies
- streamlit
- pandas
- numpy
- joblib
- wordcloud
- matplotlib
- nltk
- scikit-learn

## Features in Detail

- **Manual Input:** Users can enter a single movie review for sentiment analysis.
- **CSV Upload:** Batch processing of reviews from a CSV file.
- **Word Cloud:** Visual representation of frequently used words in reviews.
- **VADER Sentiment Analysis:** Provides compound sentiment scores.
- **Confidence Histogram:** Displays the distribution of model confidence for batch predictions.

## Future Improvements

- Implement more advanced NLP techniques like BERT or transformers for potentially better accuracy
- Add support for multi-class sentiment analysis (e.g., very negative, negative, neutral, positive, very positive)

## Contributing
There is room for contributions to improve the Movie Review Sentiment Analysis project! Here's how you can contribute:

1. Fork the repository
2. Create a new branch (git checkout -b feature-branch)
3. Make your changes and commit them (git commit -am 'Add some feature')
4. Push to the branch (git push origin feature-branch)
4. Create a new Pull Request

Please ensure your code adheres to our coding standards and include tests for new features.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions or feedback about this project, please reach out to:

- Name: Chiezie Arthur Ezenwaegbu
- Email: Chiezie.arthur@gmail.com
- GitHub: @ArthurTechy
- LinkedIn: www.linkedin.com/in/chiezie-arthur-ezenwaegbu

You can also open an issue in the GitHub repository if you encounter any problems or have feature requests.