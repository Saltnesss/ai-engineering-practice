from dotenv import load_dotenv
load_dotenv()
import os

key = os.getenv("openai_api_key")
if key:
    print(f"✅ 读取成功，key 开头是: {key[:10]}...")
else:
    print("❌ 没读到，检查 .env 文件位置或内容")