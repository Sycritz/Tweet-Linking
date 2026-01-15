import lmdb
import os
import sys
import pandas as pd

# Fix for Protobuf 4.x+ compatibility
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Add Provided-Resources to path to import pb2 files
sys.path.append(os.path.join(os.getcwd(), 'Provided-Resources'))

try:
    import SerializedListNew_pb2
    import DictionaryWithTitle_pb2
except ImportError as e:
    print(f"Error importing Protocol Buffers definition: {e}")
    sys.exit(1)

def test_inverted_index():
    print("\n" + "="*50)
    print("TESTING INVERTED INDEX (PostingsLast)")
    print("="*50)
    
    # Path should be the DIRECTORY, not the .mdb file
    db_path = os.path.join(os.getcwd(), 'Provided-Resources', 'PostingsLast')
    
    if not os.path.exists(db_path):
        print(f"❌ Path not found: {db_path}")
        return

    try:
        env = lmdb.open(db_path, readonly=True, lock=False)
        with env.begin() as txn:
            print(f"✅ Effectively opened LMDB at {db_path}")
            
            # Test 1: Search for a specific term
            term = "data mining"
            print(f"🔎 Searching for term: '{term}'")
            val = txn.get(term.encode('utf-8'))
            
            if val:
                my_list = SerializedListNew_pb2.SerializedListNew()
                my_list.ParseFromString(val)
                print(f"   found {len(my_list.Elements)} docs for '{term}'")
                print(f"   Top 3 docs: {[f'DocID: {e.docId} (Score: {e.score})' for e in my_list.Elements[:3]]}")
            else:
                print(f"   Term '{term}' not found in index.")

            # Test 2: Iterate first 5 items
            print("\n👀 Peeking at first 5 items in DB:")
            with txn.cursor() as cursor:
                count = 0
                for key, value in cursor:
                    if count >= 5: break
                    try:
                        k_str = key.decode('utf-8')
                    except:
                        k_str = key
                    
                    # Parse value
                    try:
                        my_list = SerializedListNew_pb2.SerializedListNew()
                        my_list.ParseFromString(value)
                        v_str = f"{len(my_list.Elements)} elements"
                    except:
                        v_str = "Could not parse"
                        
                    print(f"   Key: {k_str} -> Value: {v_str}")
                    count += 1

    except Exception as e:
        print(f"❌ Error accessing Inverted Index: {e}")

def test_page_context():
    print("\n" + "="*50)
    print("TESTING PAGE CONTEXTS (PageIdToContexte2)")
    print("="*50)
    
    db_path = os.path.join(os.getcwd(), 'Provided-Resources', 'PageIdToContexte2')
    
    if not os.path.exists(db_path):
        print(f"❌ Path not found: {db_path}")
        return

    try:
        env = lmdb.open(db_path, readonly=True, lock=False)
        with env.begin() as txn:
            print(f"✅ Effectively opened LMDB at {db_path}")
            
            # Test iterate
            print("\n👀 Peeking at first 3 pages:")
            with txn.cursor() as cursor:
                count = 0
                for key, value in cursor:
                    if count >= 3: break
                    
                    try:
                        page_id = key.decode('utf-8')
                        page_data = DictionaryWithTitle_pb2.Dico()
                        page_data.ParseFromString(value)
                        
                        print(f"   PageID: {page_id}")
                        print(f"   Title: {page_data.PageTitle}")
                        print(f"   PageRank: {page_data.PageRank}")
                        print(f"   Calls: {len(page_data.CalledPages)}")
                        print(f"   Categories: {list(page_data.Categories.keys())[:3]}")
                    except Exception as e:
                        print(f"   Error parsing item: {e}")
                    
                    count += 1
    except Exception as e:
         print(f"❌ Error accessing Page Contexts: {e}")

def test_meij_dataset():
    print("\n" + "="*50)
    print("TESTING MEIJ DATASET (TSV)")
    print("="*50)
    
    tsv_path = os.path.join(os.getcwd(), 'Provided-Resources', 'Datasets', 'MeijRevisedAugmented', 'MeijTweets.tsv')
    
    if os.path.exists(tsv_path):
        print(f"✅ Found Meij dataset at {tsv_path}")
        try:
            # Try reading first few lines
            df = pd.read_csv(tsv_path, sep='\t', header=None, names=['TweetID', 'User', 'Text'], quoting=3)
            print(f"   Loaded {len(df)} tweets.")
            print("\n   First 3 tweets:")
            print(df.head(3))
        except Exception as e:
            print(f"❌ Error reading TSV: {e}")
    else:
        print(f"❌ Meij dataset not found at {tsv_path}")

if __name__ == "__main__":
    test_inverted_index()
    test_page_context()
    test_meij_dataset()
