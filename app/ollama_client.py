import os

import httpx


OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://172.31.224.1:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "phi3",
)


async def generate_response(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
        )

        response.raise_for_status()

        data = response.json()
        return data["response"]