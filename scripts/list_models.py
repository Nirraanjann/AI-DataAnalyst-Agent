import os
from dotenv import load_dotenv
load_dotenv()
import requests

api_key = os.environ["GEMINI_API_KEY"]
resp = requests.get(
    f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
)
print(resp.status_code)
for m in resp.json().get("models", []):
    if "generateContent" in m.get("supportedGenerationMethods", []):
        print(m["name"])