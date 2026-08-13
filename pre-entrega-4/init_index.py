"""Verificación/creación del índice Serverless de la pre-entrega 4 (Fase 4).

init_index() es idempotente (RF-1): si el índice config.INDEX_NAME no existe,
lo crea con la spec Serverless (cloud/región de config, config.DIMENSION,
métrica cosine) y hace poll hasta el estado READY; si existe, verifica
dimensión/métrica (advertencia si difieren) y continúa sin recrear. Sin
PINECONE_API_KEY sale con mensaje claro antes de tocar la red (SystemExit
desde main).
"""

from __future__ import annotations

import sys
import time

import pinecone

from config import (
    DIMENSION,
    INDEX_NAME,
    METRIC,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_REGION,
)

# Cadencia del poll a READY y timeout total de espera (RF-1).
_POLL_INTERVALO = 10
_TIMEOUT_SEGUNDOS = 300


def init_index(
    timeout_segundos: int = _TIMEOUT_SEGUNDOS,
    poll_intervalo: int = _POLL_INTERVALO,
) -> dict:
    """Verifica o crea el índice Serverless y espera a que quede READY.

    Devuelve un resumen {"indice", "creado", "dimension", "metric", "estado"}
    para el CLI. Levanta RuntimeError si falta PINECONE_API_KEY o si el índice
    no queda READY dentro del timeout.
    """
    if not PINECONE_API_KEY:
        raise RuntimeError(
            "Falta PINECONE_API_KEY en el entorno (.env). Completá la clave de "
            "api.pinecone.io para poder crear o verificar el índice."
        )

    cliente = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    try:
        descripcion = cliente.describe_index(INDEX_NAME)
        creado = False
    except pinecone.exceptions.NotFoundException:
        # pinecone 7.3.0 (SDK v6.x): un índice inexistente lanza
        # NotFoundException (404) — NO devuelve None — y ahí se dispara la
        # creación (RF-1, idempotente).
        descripcion = None
        creado = True

    if creado:
        cliente.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=pinecone.ServerlessSpec(
                cloud=PINECONE_CLOUD, region=PINECONE_REGION
            ),
        )
    elif descripcion.dimension != DIMENSION or descripcion.metric != METRIC:
        print(
            f"[init_index] ATENCIÓN: el índice '{INDEX_NAME}' ya existe con "
            f"{descripcion.dimension} dimensiones y métrica {descripcion.metric}; "
            f"se esperaba {DIMENSION}d/{METRIC}. Recrealo si la dimensión difiere "
            f"del modelo de embeddings configurado.",
            file=sys.stderr,
        )

    inicio = time.monotonic()
    while True:
        try:
            estado = cliente.describe_index(INDEX_NAME)
        except pinecone.exceptions.NotFoundException:
            # Defensivo: el índice recién creado puede tardar en propagarse y
            # describe_index podría seguir respondiendo 404; se continúa el poll
            # hasta el timeout, que sí lanza RuntimeError.
            estado = None
        if estado is not None and estado.status.ready:
            break
        if time.monotonic() - inicio > timeout_segundos:
            raise RuntimeError(
                f"El índice '{INDEX_NAME}' no quedó READY tras "
                f"{timeout_segundos} segundos."
            )
        time.sleep(poll_intervalo)

    return {
        "indice": INDEX_NAME,
        "creado": creado,
        "dimension": DIMENSION,
        "metric": METRIC,
        "estado": "ready",
    }


def main() -> None:
    """CLI: python init_index.py — verifica o crea el índice y espera READY."""
    try:
        resumen = init_index()
    except RuntimeError as error:
        print(f"[init_index] ERROR: {error}", file=sys.stderr)
        sys.exit(1)
    accion = "creado" if resumen["creado"] else "verificado"
    print(
        f"[init_index] Índice '{resumen['indice']}' {accion} "
        f"({resumen['dimension']}d, métrica {resumen['metric']}) — "
        f"estado {resumen['estado']}."
    )


if __name__ == "__main__":
    main()
