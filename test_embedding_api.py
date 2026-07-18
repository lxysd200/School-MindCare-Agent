from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
resp = client.embeddings.create(
    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-v4"),
    input=["最近学习压力很大，总感觉睡不好。"],
    encoding_format="float",
    dimensions=1024,
)

print("model:", resp.model)
print("vector length:", len(resp.data[0].embedding))
print("first 5:", resp.data[0].embedding[:5])