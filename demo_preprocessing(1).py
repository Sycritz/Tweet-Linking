import text_preprocessing

samples = [
    "Just setting up my twttr #firsttweet",
    "Check out this amazing article: http://example.com/article123 @TechNews",
    "   Messy   whitespace   and   CAPS!   ",
    "Python 3.10 is out! #python #programming",
    "NoSpecialCharsHere",
    "I looooove coooofffeee!!!",
    "Visit www.google.com for more info."
]

print("-" * 50)
print(f"{'Original':<40} | {'Cleaned':<40} | {'Normalized':<40}")
print("-" * 50)

for text in samples:
    cleaned = text_preprocessing.clean_text(text)
    normalized = text_preprocessing.normalize_text(cleaned)
    print(f"{text[:37]+'...':<40} | {cleaned[:37]+'...':<40} | {normalized[:37]+'...':<40}")
    
    ngrams = text_preprocessing.get_ngrams(normalized, 1, 2)
    print(f"  -> N-grams (1-2): {ngrams[:5]}...")
    print("-" * 50)
