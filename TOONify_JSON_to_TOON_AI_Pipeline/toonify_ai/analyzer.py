# Structure Analyzer

def analyze_structure(data):
    """
    Analyze JSON structure for TOON suitability
    """
    result = {
        "is_array": isinstance(data, list),
        "uniform_keys": False,
        "array_length": 0,
        "key_count": 0,
        "nesting_depth": 0,
    }

    # Check if the top-level object is a list (JSON Array).
    if isinstance(data, list):
        result["array_length"] = len(data) # If data is a list, how many items are inside.

        # Checks if the first item in the list is a dictionary (JSON Object).
        if len(data) > 0 and isinstance(data[0], dict):
            keys = set(data[0].keys()) # Grab the keys from the 1st object.
            result["uniform_keys"] = all(set(d.keys()) == keys for d in data) # Check if every single object in the list has the exact same keys as the first one.
            result["key_count"] = len(keys)

    # Check how deep the nesting goes.
    def depth(obj, level=1):
        if isinstance(obj, dict):
            return max([depth(v, level + 1) for v in obj.values()], default=level)
        elif isinstance(obj, list):
            return max([depth(i, level + 1) for i in obj], default=level)
        return level

    result["nesting_depth"] = depth(data)

    return result