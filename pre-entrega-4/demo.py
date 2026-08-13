"""Demo interactiva del pipeline RAG: retrieve híbrido + generación (RF-6).

Script de demostración solicitado por el orquestador (m0144): levanta
RAGSystem con la config de config.py, recupera el top-k híbrido (RRF, con
score de posición), genera la respuesta con el proveedor LLM configurado
(LLM_PROVIDER, default gemini vía Vertex AI) e imprime todo en consola.

Uso:
    HF_HUB_OFFLINE=1 python3 pre-entrega-4/demo.py "¿Cómo defino un decorador POST en FastAPI?"
    python3 pre-entrega-4/demo.py        # sin argumento: pregunta por prompt interactivo

Imports lazy: importar este módulo NO instancia RAGSystem, embeddings ni el
chat model (todo se construye recién dentro de main(); la lección #866 —
load_dotenv no resuelve .env desde `python3 -c`— es la razón de que la demo
sea un script en archivo y no inline). El score mostrado por hit es la
contribución RRF desde el rango de fusión (peso/(c+rango), misma fórmula que
rrf_combine de rag_system): retrieve() expone los Document sin el score
crudo, y el rango de fusión es la mejor posición observable del ranking.
"""

from __future__ import annotations

import sys

from config import RRF_C, TOP_K

# Pregunta de ejemplo del corpus (decoradores de rutas en FastAPI): se usa
# cuando no se pasa argumento CLI y el prompt interactivo queda vacío.
PREGUNTA_DEFAULT = "¿Cómo defino un decorador POST en FastAPI?"


def _score_desde_rango(rango: int, c: int = RRF_C) -> float:
    """Contribución RRF de una posición: peso/(c+rango), con peso=1 (D8).

    retrieve() devuelve los Document sin el score crudo de la fusión, así
    que la demo muestra la contribución RRF del rango final: decrece con la
    posición y es estrictamente monótona respecto del orden del ranking.
    """
    return 1.0 / (c + rango)


def _obtener_pregunta(argv: list[str]) -> str:
    """Pregunta desde el argumento CLI; si no, desde el prompt interactivo.

    Prioridad: argumento de sys.argv > input() > PREGUNTA_DEFAULT (una
    pregunta del corpus para que la demo siempre pueda arrancar).
    """
    if argv:
        por_argumento = " ".join(argv).strip()
        if por_argumento:
            return por_argumento
    pregunta = input("Pregunta para el RAG (Enter usa el ejemplo): ").strip()
    return pregunta or PREGUNTA_DEFAULT


def _formatear_hits(hits) -> str:
    """Top-k como lista con rango, document_id, sección y score RRF.

    El rank 1 (mejor coincidencia de la fusión híbrida) se destaca para la
    demo; los document_id salen de la metadata de cita (D4/RF-3), nunca del
    LLM.
    """
    lineas = []
    for rango, hit in enumerate(hits, start=1):
        doc_id = hit.metadata.get("document_id", "desconocido")
        seccion = hit.metadata.get("seccion", "")
        detalle = f" (sección: {seccion})" if seccion else ""
        score = _score_desde_rango(rango)
        destacado = "  ← mejor coincidencia" if rango == 1 else ""
        lineas.append(f"  {rango}. {doc_id}{detalle} — score {score:.4f}{destacado}")
    return "\n".join(lineas)


def _imprimir_respuesta(respuesta) -> None:
    """Bloque de la respuesta generada (LlmAnswer), con degradación controlada.

    Si el LLM no pudo responder (answered=False: contexto insuficiente o
    error del proveedor), se imprime un mensaje claro sin romper el flujo
    (RF-6 edge), igual que lo hace responder() aguas arriba.
    """
    print("\n--- Respuesta generada ---")
    if not respuesta.answered:
        print("(No se pudo generar: answered=false)")
        print(respuesta.respuesta)
        print("Fuentes: ninguna (contexto insuficiente o error del proveedor).")
        return
    print(respuesta.respuesta)
    fuentes = ", ".join(respuesta.fuentes) if respuesta.fuentes else "(sin fuentes)"
    print(f"Fuentes: {fuentes}")


def main(argv: list[str] | None = None) -> int:
    """Corre la demo end-to-end: retrieve top-k + responder, en consola.

    - pregunta: argumento CLI o prompt interactivo (con default del corpus).
    - top-k: retrieve() híbrido (BM25 + Pinecone, RRF) con dedupe a nivel
      documento; se imprime con document_id/score y el rank 1 destacado.
    - respuesta: responder() -> LlmAnswer; si answered=False se imprime la
      degradación sin crash (exit code 0: el fallo controlado no es un error
      de la demo).

    RAGSystem se importa dentro de main() (lazy, patrón de evaluate.py):
    importar demo.py no construye embeddings ni el chat model.
    """
    from rag_system import RAGSystem

    args = list(sys.argv[1:] if argv is None else argv)
    pregunta = _obtener_pregunta(args)
    print(f"Pregunta: {pregunta}\n")

    rag = RAGSystem()
    hits = rag.retrieve(pregunta, k=TOP_K)
    print("--- Top 5 recuperado (RRF híbrido, a nivel documento) ---")
    print(_formatear_hits(hits) if hits else "  (sin coincidencias)")

    _imprimir_respuesta(rag.responder(pregunta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
