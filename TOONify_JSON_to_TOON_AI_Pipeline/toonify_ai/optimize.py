# The Recommender Engine

import json
from .converter import json_to_toon
from .tokenizer import estimate_tokens
from .recommender import recommend_format

def optimize(data, model="gpt-4"):
    json_str = json.dumps(data, separators=(",", ":"))

    recommendation = recommend_format(data)

    result = {
        "recommended_format": recommendation["recommended_format"],
        "reason": recommendation["reason"],
    }

    json_tokens = estimate_tokens(json_str, model)
    result["json_tokens"] = json_tokens

    if recommendation["recommended_format"] == "TOON":
        try:
            toon_str = json_to_toon(data)
            toon_tokens = estimate_tokens(toon_str, model)

            result["toon_tokens"] = toon_tokens
            result["savings_percent"] = round((json_tokens - toon_tokens) / json_tokens * 100, 2)
            result["output"] = toon_str
        except Exception as e:
            result["error"] = str(e)

    return result