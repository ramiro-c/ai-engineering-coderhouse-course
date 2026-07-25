from __future__ import annotations

import asyncio

from chain import process_text
from schemas import ExtractionError

TEXTO_ARQUITECTURA = (
    "El servicio de pagos expone una API construida con FastAPI. Las sesiones "
    "de usuario se cachean en Redis y las transacciones persisten en PostgreSQL. "
    "Bajo carga concurrente el pool de conexiones a PostgreSQL se agota y las "
    "requests empiezan a devolver timeouts."
)

TEXTO_AMBIGUO = (
    "¿Algo se rompió?"
)

TEXTO_SIN_INFORMACION_TECNICA = (
    "hola"
)

async def run(texto: str) -> None:
    resultado = await process_text(texto)
    print(resultado.model_dump_json(indent=2))
    if isinstance(resultado, ExtractionError):
        print("(el pipeline devolvió un error, no una extracción)")


async def main() -> None:
    print("=== Caso 1: descripción de arquitectura ===")
    await run(TEXTO_ARQUITECTURA)

    print("\n=== Caso 2: texto ambiguo ===")
    await run(TEXTO_AMBIGUO)

    print("\n=== Caso 3: texto sin información técnica ===")
    await run(TEXTO_SIN_INFORMACION_TECNICA)


if __name__ == "__main__":
    asyncio.run(main())
