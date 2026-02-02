"""Easy-to-use inference interface for entity linking.

Provides a simple API to link entities in tweets using any trained model.
"""

import os
import sys
    
    parser = argparse.ArgumentParser(description="Link entities in a tweet")
    parser.add_argument("--tweet", type=str, required=True, help="Tweet text")
    parser.add_argument("--model", type=str, default="models/svm_entity_linker.pkl",
                        help="Path to model file")
    parser.add_argument("--model-type", type=str, default="svm",
                        choices=["svm", "xgboost", "dnn"])
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    
    with EntityLinker(args.model, args.model_type) as linker:
        print(f"\nTweet: {args.tweet}")
        print("-" * 50)
        
        results = linker.link_entities(args.tweet, threshold=args.threshold)
        
        if results:
            print("Linked Entities:")
            for r in results:
                print(f"  {r}")
        else:
            print("No entities linked above threshold.")


if __name__ == "__main__":
    main()
