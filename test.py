import openai
from openai import OpenAI

client = OpenAI(
    api_key="key"
)

try:
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "user", "content": "Hello, are you working?"}
        ]
    )
    print("✅ OpenAI API key is valid.")
    print("Response:", response.choices[0].message.content)
except Exception as e:
    print("❌ OpenAI key invalid or request failed.")
    print("Error:", e)
