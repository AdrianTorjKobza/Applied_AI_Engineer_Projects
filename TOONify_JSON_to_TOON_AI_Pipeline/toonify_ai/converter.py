# JSON to TOON Conversion

def is_uniform_array(data):
    if not isinstance(data, list) or len(data) == 0:
        return False
    
    first_keys = set(data[0].keys())
    return all(set(item.keys()) == first_keys for item in data)


def json_to_toon(data, root_name="data"):
    """
    Convert JSON array of objects into TOON format.
    """
    if not is_uniform_array(data):
        raise ValueError("TOON conversion requires a uniform array of objects.")

    keys = list(data[0].keys())
    header = f"{root_name}[{len(data)}]{{{','.join(keys)}}}:"

    rows = []

    for item in data:
        row = ",".join(str(item[k]) for k in keys)
        rows.append(f"  {row}")

    return header + "\n" + "\n".join(rows)