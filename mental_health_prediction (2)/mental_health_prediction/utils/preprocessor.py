"""
Text Preprocessing Module
Cleans and prepares textual data for analysis.
"""

import re
import string

# Common English stop words (inline to avoid NLTK dependency)
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
    'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its',
    'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
    'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
    'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
    "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
    "ain't", "aren't", "couldn't", "didn't", "doesn't", "hadn't", "hasn't", "haven't",
    "isn't", "mightn't", "mustn't", "needn't", "shan't", "shouldn't", "wasn't",
    "weren't", "won't", "wouldn't"
}


def clean_text(text: str) -> str:
    """Full preprocessing pipeline for a single text input."""
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove mentions and hashtags symbols (keep word)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize (simple whitespace split)
    tokens = text.split()

    # Remove stop words and short tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    return ' '.join(tokens)


def get_sentiment_score(text: str) -> dict:
    """
    Rule-based sentiment scoring using keyword lexicons.
    Returns scores for positive, negative, and neutral sentiments.
    """
    positive_words = {
        'happy', 'joy', 'love', 'great', 'good', 'wonderful', 'amazing', 'excited',
        'hopeful', 'grateful', 'blessed', 'fantastic', 'excellent', 'positive',
        'cheerful', 'delighted', 'peaceful', 'energetic', 'motivated', 'optimistic',
        'confident', 'proud', 'calm', 'content', 'satisfied', 'thrilled', 'glad'
    }
    negative_words = {
        'sad', 'depressed', 'anxious', 'stressed', 'hopeless', 'worthless', 'tired',
        'exhausted', 'lonely', 'empty', 'numb', 'broken', 'lost', 'miserable',
        'overwhelmed', 'scared', 'panic', 'fear', 'hate', 'angry', 'frustrated',
        'crying', 'hurt', 'pain', 'suffering', 'desperate', 'helpless', 'failed',
        'failure', 'useless', 'burden', 'dark', 'dead', 'die', 'death', 'kill'
    }

    words = set(text.lower().split())
    pos_count = len(words & positive_words)
    neg_count = len(words & negative_words)
    total = max(pos_count + neg_count, 1)

    pos_score = round(pos_count / total, 3)
    neg_score = round(neg_count / total, 3)
    neu_score = round(1 - pos_score - neg_score, 3)
    neu_score = max(0, neu_score)

    if neg_count > pos_count:
        label = 'Negative'
    elif pos_count > neg_count:
        label = 'Positive'
    else:
        label = 'Neutral'

    return {
        'positive': pos_score,
        'negative': neg_score,
        'neutral': neu_score,
        'label': label
    }
