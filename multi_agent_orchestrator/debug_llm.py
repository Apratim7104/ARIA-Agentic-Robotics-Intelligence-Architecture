import os
from llm.client import client

MODEL = os.getenv("OPENAI_MODEL", "qwen-3.6-27b")

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    max_tokens=100
)

msg = response.choices[0].message
print("=== RAW MESSAGE OBJECT ===")
print(repr(msg))
print()
print(f".content => {repr(msg.content)}")
print(f".reasoning_content => {repr(getattr(msg, 'reasoning_content', 'ATTR_NOT_FOUND'))}")
print(f".__dict__ => {msg.__dict__}")
