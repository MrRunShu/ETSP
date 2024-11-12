import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_tweet(tweet):
    if isinstance(tweet, str):
        # Lowercase the tweet
        tweet = tweet.lower()
        # Remove URLs
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        # Remove user mentions
        tweet = re.sub(r'@\w+', '', tweet)
        # Remove hashtags
        tweet = re.sub(r'#\w+', '', tweet)
        # Remove special characters and punctuation
        tweet = re.sub(r'[^\w\s]', '', tweet)
        # Convert emojis to text
        tweet = emoji.demojize(tweet)
        # Tokenize
        words = word_tokenize(tweet)
        # Remove stopwords and lemmatize words
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return " ".join(words)
    return tweet


# Example usage
tweets = [
    "no one ever predicted this was going to happen. #sarcasm https://t.co/eeJsozDYKc",
    "@Stooshie its as closely related as Andrews original claim that evolution and entropy #sarcasm",
    "I find it ironic when Vegans say they love food #Irony"
]
preprocessed_tweets = [preprocess_tweet(tweet) for tweet in tweets]

# Print preprocessed text
for text in preprocessed_tweets:
    print(f"Text: {text}\n")
