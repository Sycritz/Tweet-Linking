# candidate_generation.py
"""
Candidate Generation Module for Tweet Entity Linking System (TELS)

This module extracts n-grams from preprocessed tweets and generates 
candidate (n-gram, PageID) pairs using the inverted index.
"""

import re
import lmdb
import os
import sys

# Add the path to Provided-Resources for importing SerializedListNew_pb2
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Provided-Resources'))

import SerializedListNew_pb2
    

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'


class TextPreprocessor:
    """
    Handles tweet text preprocessing according to TELS rules.
    Rule R1: Remove Twitter user mentions (@username)
    """
    
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#(\w+)')
        
    def clean_tweet(self, text):
        """
        Clean tweet text by removing URLs, @mentions, and expanding hashtags.
        
        Args:
            text (str): Raw tweet text
            
        Returns:
            str: Cleaned tweet text
        """
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Remove Twitter mentions (Rule R1)
        text = self.mention_pattern.sub('', text)
        
        # Expand hashtags (extract camelCase words)
        def expand_hashtag(match):
            hashtag = match.group(1)
            # Split on capital letters for camelCase
            expanded = re.sub(r'([A-Z])', r' \1', hashtag).strip()
            return expanded
        
        text = self.hashtag_pattern.sub(expand_hashtag, text)
        
        # Clean extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def split_into_phrases(self, text):
        """
        Split text into phrases at punctuation marks.
        This prevents extraction of n-grams that cross punctuation boundaries.
        
        Args:
            text (str): Cleaned tweet text
            
        Returns:
            list: List of phrases
        """
        # Handle acronyms by protecting dots in them
        text = re.sub(r'(\b[A-Z]\.)+', lambda m: m.group(0).replace('.', '<!DOT!>'), text)
        
        # Split on punctuation: . , ! ? ; :
        phrases = re.split(r'[.,!?;:]', text)
        
        # Restore dots in acronyms
        phrases = [p.replace('<!DOT!>', '.').strip() for p in phrases if p.strip()]
        
        return phrases


class NgramExtractor:
    """
    Extracts n-grams from text phrases.
    """
    
    def __init__(self, max_ngram_length=6):
        """
        Args:
            max_ngram_length (int): Maximum n-gram length (default: 6 per TELS paper)
        """
        self.max_ngram_length = max_ngram_length
    
    def extract_ngrams(self, phrases):
        """
        Extract all n-grams from phrases.
        
        Args:
            phrases (list): List of text phrases
            
        Returns:
            list: List of tuples (ngram, phrase_id, start_position, length)
        """
        ngrams = []
        
        for phrase_id, phrase in enumerate(phrases):
            tokens = phrase.split()
            num_tokens = len(tokens)
            
            # Extract n-grams of varying lengths
            for n in range(1, min(num_tokens + 1, self.max_ngram_length + 1)):
                for start_pos in range(num_tokens - n + 1):
                    ngram_tokens = tokens[start_pos:start_pos + n]
                    ngram = ' '.join(ngram_tokens)
                    
                    ngrams.append({
                        'ngram': ngram.lower(),  # Lowercase for matching
                        'original': ngram,        # Keep original case
                        'phrase_id': phrase_id,
                        'start_pos': start_pos,
                        'length': n
                    })
        
        return ngrams


class InvertedIndexAccess:
    """
    Wrapper for accessing the inverted index (PostingsLast LMDB).
    """
    
    def __init__(self, index_path):
        """
        Args:
            index_path (str): Path to PostingsLast LMDB database
        """
        self.index_path = index_path
        self.env = lmdb.open(index_path, readonly=True, max_dbs=0)
    
    def get_candidates(self, ngram):
        """
        Look up an n-gram in the inverted index.
        
        Args:
            ngram (str): N-gram to look up (lowercase)
            
        Returns:
            list: List of dicts with keys: pageId, score, type
                  type: 0=redirect, 1=anchor, 2=both
        """
        candidates = []
        
        with self.env.begin() as txn:
            val = txn.get(ngram.encode())
            
            if val is not None:
                posting_list = SerializedListNew_pb2.SerializedListNew()
                posting_list.ParseFromString(val)
                
                for element in posting_list.Elements:
                    candidates.append({
                        'pageId': element.docId,
                        'score': element.score,
                        'type': element.typ  # 0=redirect, 1=anchor, 2=both
                    })
        
        return candidates
    
    def close(self):
        """Close the LMDB environment."""
        self.env.close()


class CandidateGenerator:
    """
    Main class for generating candidate (n-gram, PageID) pairs.
    """
    
    def __init__(self, index_path, max_ngram_length=6, min_score=1):
        """
        Args:
            index_path (str): Path to PostingsLast LMDB database
            max_ngram_length (int): Maximum n-gram length
            min_score (int): Minimum score threshold for filtering candidates
        """
        self.preprocessor = TextPreprocessor()
        self.ngram_extractor = NgramExtractor(max_ngram_length)
        self.index = InvertedIndexAccess(index_path)
        self.min_score = min_score
    
    def generate_candidates(self, tweet_text):
        """
        Generate all candidate (n-gram, PageID) pairs for a tweet.
        
        Args:
            tweet_text (str): Raw tweet text
            
        Returns:
            list: List of candidate dictionaries with keys:
                  - ngram: the n-gram string
                  - pageId: Wikipedia page ID
                  - score: anchor frequency score
                  - type: 0=redirect, 1=anchor, 2=both
                  - phrase_id: which phrase the n-gram came from
                  - start_pos: position in phrase
                  - length: n-gram length
        """
        # Step 1: Preprocess tweet
        cleaned_text = self.preprocessor.clean_tweet(tweet_text)
        
        # Step 2: Split into phrases
        phrases = self.preprocessor.split_into_phrases(cleaned_text)
        
        # Step 3: Extract n-grams
        ngrams = self.ngram_extractor.extract_ngrams(phrases)
        
        # Step 4: Look up each n-gram in inverted index
        candidates = []
        
        for ngram_info in ngrams:
            ngram = ngram_info['ngram']
            index_results = self.index.get_candidates(ngram)
            
            # Generate candidate pairs
            for result in index_results:
                # Filter by minimum score
                if result['score'] >= self.min_score:
                    candidate = {
                        'ngram': ngram_info['original'],  # Original case
                        'ngram_lower': ngram,             # Lowercase for matching
                        'pageId': result['pageId'],
                        'score': result['score'],
                        'type': result['type'],
                        'phrase_id': ngram_info['phrase_id'],
                        'start_pos': ngram_info['start_pos'],
                        'length': ngram_info['length']
                    }
                    candidates.append(candidate)
        
        return candidates
    
    def generate_candidates_batch(self, tweets):
        """
        Generate candidates for multiple tweets.
        
        Args:
            tweets (list): List of tweet texts
            
        Returns:
            list: List of candidate lists (one per tweet)
        """
        return [self.generate_candidates(tweet) for tweet in tweets]
    
    def close(self):
        """Clean up resources."""
        self.index.close()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    
    # Initialize candidate generator
    INDEX_PATH = 'Provided-Resources/PostingsLast'  # Path to the LMDB database
    
    generator = CandidateGenerator(
        index_path=INDEX_PATH,
        max_ngram_length=6,
        min_score=1  # Filter candidates with score < 1
    )
    
    # Example tweet
    tweet = "Alan Turing was a brilliant cryptanalyst who worked at Bletchley Park during WWII. #AI #ComputerScience"
    
    print(f"Original tweet:\n{tweet}\n")
    
    # Generate candidates
    candidates = generator.generate_candidates(tweet)
    
    print(f"Generated {len(candidates)} candidate pairs:\n")
    
    # Display first 20 candidates
    for i, candidate in enumerate(candidates[:20]):
        print(f"{i+1}. N-gram: '{candidate['ngram']}' -> PageID: {candidate['pageId']}")
        print(f"   Score: {candidate['score']}, Type: {candidate['type']}, Length: {candidate['length']}")
        print()
    
    # Statistics
    print(f"\nTotal candidates: {len(candidates)}")
    print(f"Unique n-grams: {len(set(c['ngram_lower'] for c in candidates))}")
    print(f"Unique pages: {len(set(c['pageId'] for c in candidates))}")
    
    # Clean up
    generator.close()