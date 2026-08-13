"""Demo interactiva del pipeline RAG: retrieve híbrido + generación (RF-6).

Script de demostración solicitado por el orquestador (m0144): levanta
RAGSystem con la config de config.py, recupera el top-k híbrido (RRF, con
score de posición), genera la respuesta con el proveedor LLM configurado
(LLM_PROVIDER, default gemini vía Vertex AI) e imprime todo en consola.

Uso:
    HF_HUB_OFFLINE=1 python3 pre-entrega-4/demo.py                  # REPL: varias preguntas, salir con 'salir'
    HF_HUB_OFFLINE=1 python3 pre-entrega-4/demo.py "¿Cómo defino un decorador POST en FastAPI?"  # una pregunta y termina

Sin argumento CLI, la demo entra en un modo interactivo (REPL) pensado para
uso humano natural: escribe una pregunta, Enter para consultar, y repite
hasta escribir 'salir', 'exit', 'q' (o Ctrl+C / Ctrl+D). Con argumento CLI,
responde esa única pregunta y termina (para scripts).

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
# cuando no se pasa argumento CLI.
PREGUNTA_DEFAULT = "¿Cómo defino un decorador POST en FastAPI?"

# Comandos de salida del REPL (case-insensitive, con espacios alrededor).
_COMANDOS_SALIDA = {"salir", "exit", "quit", "q", "s"}


def _score_desde_rango(rango: int, c: int = RRF_C) -> float:
    """Contribución RRF de una posición: peso/(c+rango), con peso=1 (D8).

    retrieve() devuelve los Document sin el score crudo de la fusión, así
    que la demo muestra la contribución RRF del rango final: decrece con la
    posición y es estrictamente monótona respecto del orden del ranking.
    """
    return 1.0 / (c + rango)


def _obtener_pregunta(argv: list[str]) -> str:
    """Pregunta desde el argumento CLI; si no, la del corpus.

    Prioridad: argumento de sys.argv > PREGUNTA_DEFAULT. El input
    interactivo vive en el REPL (_modo_interactivo), no aquí.
    """
    por_argumento = " ".join(argv).strip()
    return por_argumento or PREGUNTA_DEFAULT


def _es_comando_salida(texto: str) -> bool:
    """¿El texto es un comando de salida del REPL? (salir/exit/q/s...)."""
    return texto.strip().lower() in _COMANDOS_SALIDA


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


def _correr_una_pregunta(rag, pregunta: str) -> None:
    """Consulta completa para una pregunta: retrieve top-k + responder + salida."""
    print(f"\nPregunta: {pregunta}\n")
    hits = rag.retrieve(pregunta, k=TOP_K)
    print("--- Top 5 recuperado (RRF híbrido, a nivel documento) ---")
    print(_formatear_hits(hits) if hits else "  (sin coincidencias)")

    _imprimir_respuesta(rag.responder(pregunta))


def _modo_interactivo(rag) -> int:
    """REPL: varias preguntas seguidas hasta que el humano sale.

    - input() con prompt "Pregunta: "; Enter vacío vuelve a preguntar.
    - 'salir', 'exit', 'quit', 'q' o 's' termina (case-insensitive).
    - Ctrl+C (KeyboardInterrupt) y Ctrl+D (EOFError) terminan con despedida.
    - Nunca crashea: el error de lectura de input no rompe la demo.
    """
    print("Demo RAG pre-entrega-4 — escribí una pregunta o 'salir' para terminar.")
    while True:
        try:
            texto = input("\nPregunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            return 0
        if not texto:
            print("(Pregunta vacía — escribí algo o 'salir'.)")
            continue
        if _es_comando_salida(texto):
            print("Hasta luego.")
            return 0
        _correr_una_pregunta(rag, texto)


def main(argv: list[str] | None = None) -> int:
    """Corre la demo: una pregunta (CLI) o REPL interactivo, en consola.

    - con argumento CLI: responde esa pregunta y termina (uso scriptable).
    - sin argumento: entra al REPL _modo_interactivo (uso humano natural,
      varias preguntas, salir con 'salir'/Ctrl+C/Ctrl+D).
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
    rag = RAGSystem()
    if not args:
        return _modo_interactivo(rag)
    _correr_una_pregunta(rag, _obtener_pregunta(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
