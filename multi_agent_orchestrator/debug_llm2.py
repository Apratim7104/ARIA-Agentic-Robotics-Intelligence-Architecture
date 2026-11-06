import os
from llm.client import client

MODEL = os.getenv("OPENAI_MODEL", "qwen-3.6-27b")

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    max_tokens=100
)

msg = response.choices[0].message

# Try every possible way to get the content
print(f".content => {repr(msg.content)}")
print(f".reasoning => {repr(getattr(msg, 'reasoning', 'NOT_FOUND'))}")
print(f".reasoning_content => {repr(getattr(msg, 'reasoning_content', 'NOT_FOUND'))}")

# Dump all fields via model_dump if available
if hasattr(msg, 'model_dump'):
    print(f"\nmodel_dump() => {msg.model_dump()}")
