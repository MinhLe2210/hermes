from _pickle import load
import asyncio
import json
import aiohttp
import time
from typing import Optional
import dotenv
from google.genai import Client, types
from langgraph.graph import StateGraph, START, MessagesState

class LLM:
    def __init__(self, model_name: str, *, timeout: float = 60.0):
        dotenv.load_dotenv()
        self.client = Client()
        self.model_name = model_name
        self.timeout = timeout

    def invoke(self, prompt: str, *, temperature: float = 0.3, max_tokens: Optional[int] = None, top_p: float = 0.95, frequency_penalty: float = 0.5) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=top_p,
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    ),
                ],
            ),
        )

        return response.text


if __name__ == "__main__":
    llm = LLM(model_name="gemini-2.5-flash")
    res = llm.invoke("3 * 6")
    print(res)