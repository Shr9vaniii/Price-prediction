from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)

DEFAULT_MODEL_NAME = "groq/openai/gpt-oss-20b"
DEFAULT_REASONING_EFFORT = "low"
groq_api_key = os.getenv("GROQ_API_KEY")
groq_base_url = "https://api.groq.com/openai/v1"

groq = OpenAI(api_key=groq_api_key, base_url=groq_base_url)

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class Preprocessor:
    def __init__(
        self,
        model_name=DEFAULT_MODEL_NAME,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
        base_url=None,
    ):
        self.groq = groq
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort

    def messages_for(self, text: str) -> list[dict]:
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]

    def preprocess(self, text: str) -> str:
        messages = self.messages_for(text)
        response = groq.chat.completions.create(
            messages=messages, model=self.model_name, reasoning_effort=self.reasoning_effort
        )
        return response.choices[0].message.content
