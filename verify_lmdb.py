
import sys
import os

# Fix for Protobuf 4.x+ compatibility with older generated code
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import lmdb

# Add Provided-Resources to path so we can import the pb2 files
sys.path.append(os.path.join(os.getcwd(), 'Provided-Resources'))

try:
    import SerializedListNew_pb2
except ImportError as e:
    print(f"Error importing Protocol Buffers definition: {e}")
    sys.exit(1)

def verify_lmdb_access(db_path, query_ngram="data mining"):
    print(f"Attempting to open LMDB at: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"Error: Database path does not exist: {db_path}")
        return

    try:
        # Open the LMDB environment
        env = lmdb.open(db_path, readonly=True, lock=False)
        
        with env.begin() as txn:
            print(f"Successfully opened database. querying for: '{query_ngram}'")
            
            # Try to get data
            # The keys in the inverted index are ngrams encoded as bytes
            val = txn.get(query_ngram.encode('utf-8'))
            
            if val:
                print(f"Found entry for '{query_ngram}'!")
                
                # Parse the data using the protobuf definition
                my_list = SerializedListNew_pb2.SerializedListNew()
                my_list.ParseFromString(val)
                
                print(f"Number of elements: {len(my_list.Elements)}")
                print("First 5 elements:")
                for i, element in enumerate(my_list.Elements[:5]):
                    print(f"  {i+1}. DocID: {element.docId}, Score: {element.score}")
            else:
                print(f"No entry found for '{query_ngram}'. This might be expected if the ngram isn't in the index.")
                
                # Let's try to just get the first key in the db to prove it works
                cursor = txn.cursor()
                if cursor.first():
                    print(f"\nVerification Successful: Database is readable.")
                    k, v = cursor.item()
                    try:
                        key_str = k.decode('utf-8')
                        print(f"First key in DB: {key_str}")
                    except:
                        print(f"First key in DB: {k} (binary)")
                else:
                    print("\nDatabase appears to be empty.")

    except Exception as e:
        print(f"An error occurred while accessing the database: {e}")

if __name__ == "__main__":
    # Path to the PostingsLast directory
    lmdb_path = os.path.join(os.getcwd(), 'Provided-Resources', 'PostingsLast')
    verify_lmdb_access(lmdb_path)
