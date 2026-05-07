# The Recommender: JSON or TOON

from .analyzer import analyze_structure

def recommend_format(data):
    analysis = analyze_structure(data)

    if (
        analysis["is_array"]
        and analysis["uniform_keys"]
        and analysis["array_length"] > 3
        and analysis["nesting_depth"] <= 3
    ):
        return {
            "recommended_format": "TOON",
            "reason": "Uniform array with repeated keys",
        }

    return {
        "recommended_format": "JSON",
        "reason": f"Irregular or deeply nested structure (Depth: {analysis['nesting_depth']})",
    }