from dotenv import load_dotenv
load_dotenv()

import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    messages=[{"role": "user", "content": "What is a neural network in one sentence?"}]
)

print(response.content[0].text)