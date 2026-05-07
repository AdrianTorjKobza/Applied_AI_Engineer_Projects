from toonify_ai.optimize import optimize

data = [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
    {"id": 3, "name": "Charlie"},
]

result = optimize(data)

print(result)