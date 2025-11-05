from openai import OpenAI
import httpx

HF_TOKEN = "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"

# Create an HTTPX client with SSL verification disabled
httpx_client = httpx.Client(verify=False)

# Pass this client to OpenAI so it uses it internally
client = OpenAI(
    api_key=HF_TOKEN,
    base_url="https://router.huggingface.co/v1",
    http_client=httpx_client
)

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct:hyperbolic",
    messages=[{"role": "user", "content": "Hello"}]
)

print(completion.choices[0].message.content)