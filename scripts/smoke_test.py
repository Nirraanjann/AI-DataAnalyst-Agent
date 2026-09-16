from dotenv import load_dotenv
load_dotenv()

from app.services.llm_client import create_message

response = create_message(
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
)

for block in response.content:
    print(block)