import requests

res = requests.post(
    "http://127.0.0.1:5050/recommend",
    json={
        "query": "bloating nausea stomach pain",
        "top_n": 5
    }
)

print(res.json())