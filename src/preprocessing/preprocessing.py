import re

def clean_tweet(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'http\s+|www\.\s+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def get_ngrams(text: str, min_n: int = 1, max_n: int = 6) -> list[str]:
    if not text:
        return []
    words = text.split()
    ngrams = []
    for n in range(min_n, max_n + 1):
        if len(words) < n:
            continue
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i + n]))
    return ngrams

