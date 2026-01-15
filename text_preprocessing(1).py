import re

def clean_text(text):
    """
    Cleans text by removing URLs, @mentions, and hashtags.
    
    Args:
        text (str): The input text (e.g., a tweet).
        
    Returns:
        str: The cleaned text.
    """
    if not text:
        return ""
        
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove non-alphanumeric characters (except spaces)
    text = re.sub(r'[^\w\s]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_text(text):
    """
    Normalizes text by lowercasing and stripping whitespace.
    
    Args:
        text (str): The input text.
        
    Returns:
        str: The normalized text.
    """
    if not text:
        return ""
    return text.lower().strip()

def get_ngrams(text, min_n=1, max_n=6):
    """
    Generates n-grams from text.
    
    Args:
        text (str): The input text.
        min_n (int): Minimum n-gram size.
        max_n (int): Maximum n-gram size.
        
    Returns:
        list: A list of n-gram strings.
    """
    if not text:
        return []
        
    words = text.split()
    all_ngrams = []
    
    for n in range(min_n, max_n + 1):
        if len(words) < n:
            continue
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        all_ngrams.extend(ngrams)
        
    return all_ngrams
