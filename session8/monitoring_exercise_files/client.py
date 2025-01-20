import requests

url = "http://127.0.0.1:8000/predict"
review = "This is rye bread. I prefer french bread."
print(review)
response = requests.post(url, json={"review": review})
print(response.json())
