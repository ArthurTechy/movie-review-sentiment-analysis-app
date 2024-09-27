# importing required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import os


@st.cache_resource
def download_nltk_data():
    # Define the path where NLTK data should be stored
    nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
    
    # Check if NLTK data already exists
    if not os.path.exists(nltk_data_path):
        # If it doesn't exist, download the data
        for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
            nltk.download(resource, quiet=True)
    
    # Set the NLTK data path
    nltk.data.path.append(nltk_data_path)

# Call the function at the start of your script
download_nltk_data()

# Function to load the model and vectorizer
@st.cache_resource
def load_model():
    try:
        model = joblib.load('logistic_regression_model.pkl')
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        return model, tfidf
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Predicting sentiments for given reviews
@st.cache_data
def predict_sentiments(reviews):
    reviews_tfidf = tfidf.transform(reviews)
    predictions = model.predict(reviews_tfidf)
    return [sentiment_label[p] for p in predictions]

# Generating word cloud from text
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Performing VADER sentiment analysis
def vader_sentiment_analysis(text):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)

# function for confidence histogram
def plot_confidence_histogram(df):
    # the figure setting
    plt.figure(figsize=(10, 6))
    
    # histogram using matplotlib
    plt.hist(df, bins=20, color='blue', alpha=0.7, edgecolor='black')

    # titles and labels
    plt.title('Sentiment Confidence Histogram', fontsize=16)
    plt.xlabel('Confidence Level', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    # Show in Streamlit
    st.pyplot(plt)


# Instantiate Lemmatizer
lemmatizer = WordNetLemmatizer()
# Function to clean the text
def clean_text(text):
    try:
        # Convert to lowercase
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters
        text = re.sub(r'[^a-z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        # Join tokens back into string
        return ' '.join(tokens)
    except Exception as e:
        st.error(f"Error in clean_text: {str(e)}")
        return text # Return original text if cleaning fails

# Main app
st.title("Movie Review Sentiment Analysis")
st.write("Analyze the sentiment of movie reviews using a binary classification Logistic Regression model.")

# Loading model and define sentiment labels
model, tfidf = load_model()
sentiment_label = {0: "Negative", 1: "Positive"}

if model is not None and tfidf is not None:
    # Sidebar information
    st.sidebar.header("About")
    st.sidebar.write("""
    This application, by Arthur C.E, predicts movie review sentiments using a binary classification Logistic Regression model. <br><br> You can analyze reviews either by manually entering text or uploading a CSV file. Visual insights such as word clouds, VADER sentiment scores, and feature importance are provided. """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")  # Add a separator
    st.sidebar.header("Key Insight:")

    # Input method selection
    input_method = st.radio(
        "Choose input method",
        ("Manual Input", "CSV Upload"),
        help="For CSV upload, ensure your file has a 'review' column."
    )

    if input_method == "Manual Input":
        # Manual input processing
        review_input = st.text_area("Enter a movie review", "")
        
        if st.button("Predict Sentiment"):
            if review_input:
                cleaned_review = clean_text(review_input)
                sentiment = predict_sentiments([cleaned_review])[0]
                probability = model.predict_proba(tfidf.transform([cleaned_review]))[0]
                confidence = float(f'{probability.max():.2f}')
                
                # Update sidebar with key insight
                st.sidebar.markdown(f"**Sentiment:** {sentiment}")
                st.sidebar.markdown(f"**Confidence:** {probability.max():.2f}")
                
                # Display word cloud and VADER scores
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Word Cloud")
                    st.pyplot(generate_wordcloud(cleaned_review))
                
                with col2:
                    st.subheader("VADER Sentiment Scores")
                    vader_scores = vader_sentiment_analysis(cleaned_review)
                    binary_scores = {
                        'Positive': vader_scores['pos'],
                        'Negative': vader_scores['neg']
                    }
                    st.write(binary_scores)

                    # Create matplotlib figure
                    fig, ax = plt.subplots()
                    colors = ['green', 'red']
                    ax.bar(binary_scores.keys(), binary_scores.values(), color=colors)
                    ax.set_ylabel('Score')
                    ax.set_title('VADER Sentiment Scores')

                    # Rotate x-axis labels
                    plt.xticks(rotation=45, ha='right')

                    # Adjust layout and display the plot
                    plt.tight_layout()
                    st.pyplot(fig) 
                
                # adding space
                st.write("")

                # Predicted Sentiment and confidence score/progress display
                st.subheader("Predicted Sentiment and Confidence Score")
                st.write(f"Predicted Sentiment: **{sentiment}**")
                st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
                # Display the confidence score using a progress bar
                st.progress(confidence)

            else:
                st.error("Please enter a review.")

    elif input_method == "CSV Upload":
        # CSV upload processing
        st.info("Please ensure your CSV file has a 'review' column.")
        uploaded_file = st.file_uploader("Upload a CSV file containing a 'review' column", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'review' in df.columns:
                    # Cleaning the reviews
                    df['cleaned_review'] = df['review'].apply(clean_text)
                    
                    reviews = df['cleaned_review'].tolist()
                    sentiments = predict_sentiments(reviews)
                    probabilities = model.predict_proba(tfidf.transform(reviews))
                    
                    # Display results
                    results_df = pd.DataFrame({
                        'Review': df['review'],
                        'Predicted Sentiment': sentiments,
                        'Confidence': probabilities.max(axis=1)
                    })
                    st.write("Part of Reviews and Predicted Sentiments:")
                    st.write(results_df.head(7))
                    
                    # Generating insights for all reviews
                    all_reviews_text = ' '.join(reviews)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Word Cloud for All Reviews")
                        st.pyplot(generate_wordcloud(all_reviews_text))
                    
                    with col2:
                        st.subheader("VADER Sentiment Scores")
                        vader_scores = vader_sentiment_analysis(all_reviews_text)
                        binary_scores = {
                            'Positive': vader_scores['pos'],
                            'Negative': vader_scores['neg']
                        }
                        st.write(binary_scores)

                        # Creating matplotlib figure
                        fig, ax = plt.subplots()
                        colors = ['green', 'red']
                        ax.bar(binary_scores.keys(), binary_scores.values(), color=colors)
                        ax.set_ylabel('Score')
                        ax.set_title('VADER Sentiment Scores')

                        # Rotate x-axis labels
                        plt.xticks(rotation=45, ha='right')

                        # Adjust layout and display the plot
                        plt.tight_layout()
                        st.pyplot(fig) 
                    
                    # adding space
                    st.write("")

                    st.subheader("Sentiment Confidence Analysis")

                    st.subheader("-Distribution of Confidence Levels")
                    st.markdown("""
                    This plot shows the distribution of confidence levels for the sentiment predictions.
                    A higher concentration of confidence levels near 1 indicates strong model reliability.
                    """)

                    # Calling the function to plot the confidence histogram
                    plot_confidence_histogram(results_df['Confidence'])

                    # Updating sidebar with key insight
                    sentiment_distribution = results_df['Predicted Sentiment'].value_counts()
                    most_common_sentiment = sentiment_distribution.idxmax()
                    average_confidence = results_df['Confidence'].mean()
                    
                    st.sidebar.markdown(f"**-Most Common Sentiment:** {most_common_sentiment}")
                    st.sidebar.markdown(f"**-Average Confidence:** {average_confidence:.2f}")
                    st.sidebar.markdown(f"**-Total Reviews:** {len(reviews)}")
                else:
                    st.error("The uploaded file does not contain a 'review' column.")
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
else:
    st.error("Failed to load the model. Please check your model files and try again.")