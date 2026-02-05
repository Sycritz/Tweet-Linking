import re

def clean_tweet(text: str) -> str:
    if not text:
        return ""
    
    # Split CamelCase (e.g., #TheForceAwakens -> The Force Awakens)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Remove URLS, mentions, special chars, and extra whitespace
    text = re.sub(r'http\s+|www\.\s+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def get_ngrams(text: str, min_n: int = 1, max_n: int | None = None) -> list[str]:

    if not text:
        return []
    words = text.split()
    if max_n is None:
        max_n = len(words)
    ngrams = []
    for n in range(min_n, max_n + 1):
        if len(words) < n:
            continue
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i + n]))
    return ngrams

