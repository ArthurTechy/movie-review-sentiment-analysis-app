# Movie Review Sentiment Analysis

## Overview
This project implements a binary sentiment classification model for movie reviews using the Stanford Large Movie Review Dataset. The application, built with Streamlit, allows users to analyze the sentiment of movie reviews either through manual input or by uploading a CSV file.

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
- **Link**: [https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
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
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
```

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

## Dependencies
- streamlit
- pandas
- numpy
- joblib
- wordcloud
- matplotlib
- nltk
- scikit-learn
- shap

## Future Improvements

- Implement more advanced NLP techniques like BERT or transformers for potentially better accuracy
- Add support for multi-class sentiment analysis (e.g., very negative, negative, neutral, positive, very positive)
- Create a mobile app version for on-the-go sentiment analysis

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