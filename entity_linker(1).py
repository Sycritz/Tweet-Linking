import csv
import lmdb
import collections
import InvertedIndexAccess
import InterrogatePageIdToContextLmdb
import text_preprocessing

# --- Entity Linking Logic ---

def link_entities(tweet_text, postings_env, context_env):
    # Step 1: Clean
    cleaned_text = text_preprocessing.clean_text(tweet_text)
    
    # Step 2: Normalize (Optional, but good for index matching if index is lowercased)
    # Let's normalize it as part of the process or just use cleaned_text if index is mixed case.
    # The previous code didn't explicitly lower(), but let's try to match better.
    # For now, let's keep it consistent with previous: clean mostly. 
    # But usually indices are lowercased. Let's try to use the normalized version for n-grams.
    normalized_text = text_preprocessing.normalize_text(cleaned_text)
    
    # Get n-grams from normalized text? Or cleaned text?
    # If the index has "Obama", and we normalize to "obama", we might miss it if index is case-sensitive.
    # Looking at existing code, it didn't lower(). Let's stick to using cleaned_text for now to be safe, 
    # BUT the user asked for normalization in the prompt.
    # Let's use normalized text because that's the "module" way.
    # If this breaks, we can revert to cleaned_text.
    
    ngrams = text_preprocessing.get_ngrams(cleaned_text, 1, 6)
    
    found_entities = {} # Map ngram -> List of (Title, Score)

    with postings_env.begin() as txn_postings:
        with context_env.begin() as txn_context:
            for ngram in ngrams:
                # Query index
                # InvertedIndexAccess.IndexAccess returns (SerializedListNew, TotalOccur)
                try:
                    # We need to pass the transaction object expected by the library
                    # InvertedIndexAccess.IndexAccess expects 'txn'
                    post_list, total_occur = InvertedIndexAccess.IndexAccess(ngram, txn_postings)
                    
                    if post_list and post_list.Elements:
                        candidates = []
                        # Take top 5 candidates
                        for i, element in enumerate(post_list.Elements):
                            if i >= 5: break
                            doc_id = str(element.docId)
                            score = element.score
                            
                            # Get Page Details
                            # InterrogatePageIdToContextLmdb.get_Element returns:
                            # Title, anchors_len, views, rank, OrgAnchors, Anchors, Categories, CalledPages
                            try:
                                details = InterrogatePageIdToContextLmdb.get_Element(doc_id, txn_context)
                                title = details[0]
                                candidates.append((title, score))
                            except Exception as e:
                                # Page might not exist in context DB
                                continue
                        
                        if candidates:
                            found_entities[ngram] = candidates
                except Exception as e:
                    # Ignore errors for specific ngrams
                    pass
    
    return cleaned_text, found_entities

def main():
    movies_path = 'Datasets/MeijRevisedAugmented/MeijTweets.tsv'
    
    # Open Environments once
    try:
        postings_env = lmdb.open('PrebuiltDatasets/PostingsLast', readonly=True)
        context_env = lmdb.open('PrebuiltDatasets/PageIdToContexte2', map_size=16000000000, readonly=True)
    except Exception as e:
        print(f"Error opening databases: {e}")
        return

    print("Processing tweets...")
    try:
        with open(movies_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            count = 0
            for row in reader:
                if len(row) < 2: continue
                tweet_id = row[0]
                tweet_text = row[1]
                
                cleaned, entities = link_entities(tweet_text, postings_env, context_env)
                
                if entities:
                    print(f"\nTweet: {tweet_text}")
                    print(f"Cleaned: {cleaned}")
                    print("Found Entities:")
                    for ngram, candidates in entities.items():
                        print(f"  '{ngram}': {candidates}")
                
                count += 1
                if count >= 10: # Process first 10 for demo
                    break
    except FileNotFoundError:
        print(f"Dataset file not found at {movies_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        postings_env.close()
        context_env.close()

if __name__ == "__main__":
    main()
