from __future__ import annotations

import asyncio

from manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig


async def main() -> None:
    try:
        manager = AsyncLLMManager('gemini')
    except ValueError as exc:
        print(f"Error de configuración: {exc}")
        return
    messages = [
        ChatMessage(role="user", content="¿Qué es la entropía? Explicalo de forma sencilla." ),
    ]
    config = ModelConfig(
        model=manager.default_model,
        temperature=0.1,
        max_tokens=2048,
    )

    print(f"=== Proveedor: {manager.provider} ===")
    print("=== Respuesta normal ===")
    response = await manager.generate(messages, config)
    if response.error:
        print(response.error)
    else:
        print(response.text)

    print("\n=== Streaming ===")
    async for token in manager.stream(messages, config):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())

