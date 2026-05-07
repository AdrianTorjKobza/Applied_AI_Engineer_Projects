import json
import argparse
from toonify_ai.optimize import optimize

def main():
    print("CLI STARTED")

    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()

    print("FILE:", args.file)

    with open(args.file, "r") as f:
        data = json.load(f)

    print("DATA LOADED")

    result = optimize(data)

    print("\n--- Analysis ---")
    print("Recommended format:", result["recommended_format"])
    print("Reason:", result["reason"])
    print("JSON tokens:", result["json_tokens"])
    print("TOON tokens:", result["toon_tokens"])
    print("Savings:", result["savings_percent"], "%")

    print("\n--- TOON Output ---")
    print(result["output"])
    print("\n")


main()