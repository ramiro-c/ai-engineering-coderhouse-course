"""Tests unit de chunking por tokens de la pre-entrega 4 (Fase 3, sin red).

Cubren los contratos RF-2/D3/D6 de build_chunks(): chunks de 500-800 tokens
medidos con tiktoken cl100k_base, overlap ~100 tokens entre chunks
consecutivos y cabeceras h1-h3 antepuestas como contexto de seccion.
Corren sin red: solo tiktoken y los splitters locales de langchain.
"""

from __future__ import annotations

import pytest
import tiktoken
from config import DATA_DIR
from ingest import build_chunks

_ENC = tiktoken.get_encoding("cl100k_base")


def _tokens(texto: str) -> int:
    return len(_ENC.encode(texto))


def _solape_tokens(a: str, b: str) -> int:
    """Tokens del solape entre chunks consecutivos (sufijo de a == prefijo de b)."""
    ta, tb = _ENC.encode(a), _ENC.encode(b)
    mejor = 0
    for k in range(1, min(len(ta), len(tb)) + 1):
        if tb[:k] == ta[-k:]:
            mejor = k
    return mejor


# ~95 oraciones DISTINTAS de prosa tecnica: una sola seccion grande para que
# el splitter produzca varios chunks con solape medible. Las oraciones son
# unicas (sin plantillas repetidas) para que la medicion del overlap no
# encuentre coincidencias espurias.
_SENTENCIAS = [
    "FastAPI declara los parametros de ruta entre llaves en el decorador.",
    "El motor ASGI permite servir websockets en la misma aplicacion.",
    "Pydantic valida los cuerpos de peticion contra los modelos declarados.",
    "Uvicorn levanta el servidor con la bandera reload en desarrollo.",
    "Las dependencias con Depends se resuelven automaticamente por tipo.",
    "El middleware de CORS protege las llamadas entre origenes distintos.",
    "La documentacion OpenAPI se genera desde los tipos de las rutas.",
    "Los errores 422 indican que la validacion de entrada fallo.",
    "El TestClient de fastapi.testclient corre la app sin servidor.",
    "La inyeccion de dependencias separa la logica de la infraestructura.",
    "Los status_code personalizados cambian la respuesta del endpoint.",
    "El parametro tags agrupa las rutas en la documentacion interactiva.",
    "Las subdependencias permiten componer logica de negocio reutilizable.",
    "El refresh token rota en cada renovacion para limitar el riesgo.",
    "La autenticacion JWT protege los endpoints con un token firmado.",
    "El timezone se normaliza a UTC en las respuestas de la API.",
    "El rate limiter responde 429 cuando el cliente excede la cuota.",
    "El encabezado Retry-After indica cuando reintentar la peticion.",
    "Los parametros de consulta son opcionales y llevan un default.",
    "La serializacion JSON convierte los modelos a respuestas HTTP.",
    "El filtro de paginacion usa limit y offset en la consulta.",
    "Los tests de integracion ejercitan rutas reales de la aplicacion.",
    "El manejador global de excepciones devuelve JSON consistente.",
    "El schema de respuesta excluye los campos sensibles del modelo.",
    "El background task envia el email despues de responder al cliente.",
    "La subida de archivos se maneja con multipart en la ruta.",
    "El hook de arranque inicializa la conexion a la base de datos.",
    "El hook de cierre libera los recursos al detener la aplicacion.",
    "La configuracion se lee desde variables de entorno con defaults.",
    "El logger estructurado emite eventos con contexto de la peticion.",
    "El health check expone el estado del servicio en una ruta.",
    "Los errores de integracion externa se traducen a respuestas limpias.",
    "La cache de respuestas reduce la latencia de los endpoints.",
    "El etag permite validar condicionalmente las respuestas cacheadas.",
    "El middleware de logging registra la duracion de cada peticion.",
    "El parametro response_model define el contrato de salida.",
    "Las rutas dinamicas convierten el parametro al tipo declarado.",
    "El orden de definicion de rutas afecta la resolucion final.",
    "El decorador de excepcion mapea errores a codigos HTTP concretos.",
    "La documentacion de cada ruta describe su parametro obligatorio.",
    "El esquema JSON del modelo se exporta al OpenAPI generado.",
    "El websocket mantiene una conexion bidireccional con el cliente.",
    "La prueba unitaria cubre la logica pura sin tocar la red.",
    "El modelo de datos valida tipos, formatos y restricciones.",
    "El timeout de la peticion externa evita bloquear el worker.",
    "El prefijo de version v1 agrupa los endpoints publicos.",
    "El paginado por cursor evita los problemas del offset profundo.",
    "El objeto de respuesta permite construir headers personalizados.",
    "El manejador de formularios parsea los datos urlencoded.",
    "El schema de entrada valida los campos obligatorios del JSON.",
    "La ruta de descarga setea el content-disposition correcto.",
    "El parametro deprecated marca los endpoints en transicion.",
    "El middleware de sesion almacena el estado del usuario.",
    "El servidor de desarrollo recarga los cambios sin reiniciar.",
    "El TestClient comparte el ciclo de vida de la aplicacion.",
    "La funcion async evita bloquear el event loop en la peticion.",
    "El parametro de consulta con Enum restringe los valores validos.",
    "La respuesta HTTP transporta el cuerpo serializado en JSON.",
    "El modelo de salida filtra los atributos privados internos.",
    "El router de FastAPI organiza los endpoints por modulo.",
    "El prefijo del router agrupa rutas bajo un mismo recurso.",
    "El parametro dependencies aplica verificaciones previas.",
    "El contexto de la peticion expone el objeto Request completo.",
    "La validacion de path convierte el identificador al tipo exacto.",
    "El error de validacion lista el campo y el motivo del rechazo.",
    "La documentacion interactiva permite probar los endpoints.",
    "El response de streaming envia datos progresivamente al cliente.",
    "El cache de nivel HTTP evita recalcular respuestas repetidas.",
    "La estrategia de reintentos tolera fallas transitorias.",
    "El backoff exponencial espacia los reintentos automaticos.",
    "El circuito breaker corta las llamadas al servicio degradado.",
    "La observabilidad expone metricas, trazas y logs del servicio.",
    "El openapi.json documenta todos los endpoints del servicio.",
    "La restriccion de metodo devuelve 405 cuando no aplica.",
    "El parametro de cabecera viaja en la peticion sin el cuerpo.",
    "La cookie de sesion firma el identificador del usuario.",
    "El cache distribuido comparte estado entre las instancias.",
    "El retry del cliente maneja los errores 503 transitorios.",
    "La cola de tareas procesa los trabajos pesados en background.",
    "El webhook notifica al consumidor cuando el evento ocurre.",
    "El esquema de respuesta documenta el contrato de salida.",
    "El parametro de consulta opcional recibe un valor por defecto.",
    "La autenticacion basica transporta las credenciales codificadas.",
    "El bearer token se envia en la cabecera de autorizacion.",
    "El permiso del endpoint valida el rol del usuario autenticado.",
    "La politica CORS permite los origenes configurados explicitamente.",
    "El archivo subido se valida por tamano y tipo de contenido.",
    "El endpoint de estado reporta la version del despliegue.",
    "La ruta raiz devuelve el indice de recursos disponibles.",
    "El parametro de path se convierte al tipo del identificador.",
    "El error interno devuelve un mensaje generico sin detalles.",
    "El log de auditoria registra quien accedio a cada recurso.",
    "La plantilla de respuesta estandariza el formato de errores.",
    "El probe de readiness distingue arranque y disponibilidad.",
    "La migracion del esquema se ejecuta antes del despliegue.",
    "El contrato de la API fija la version del recurso publico.",
    "El rate limit se aplica por cliente usando la API key.",
    "El middleware comprime las respuestas para reducir el trafico de red.",
    "El endpoint de busqueda indexa los campos mas consultados.",
    "La autenticacion por OAuth delega la identidad en el proveedor externo.",
    "El handler de 404 devuelve el recurso no encontrado en JSON.",
    "El parametro form captura los campos de un formulario HTML.",
    "La ruta de eliminacion responde 204 sin cuerpo cuando tiene exito.",
    "El schema de entrada define el ejemplo en la documentacion.",
    "El middleware de seguridad agrega las cabeceras de proteccion.",
    "El paginado de la respuesta incluye el total de elementos.",
    "El endpoint de creacion devuelve 201 con el recurso creado.",
    "El timeout del cliente externo limita la espera de la respuesta.",
    "El manejador de errores registra el stack trace completo.",
    "La respuesta de error incluye el codigo interno del servicio.",
    "El test de la ruta verifica el status y el cuerpo esperado.",
]

_TEXTO_LARGO = "# Guia de API\n\n" + "\n\n".join(_SENTENCIAS)

# Documento corto con jerarquia h1>h2>h3 para validar la ruta antepuesta (D6).
_TEXTO_SECCIONES = (
    "# Guia de API\n\n"
    "## Autenticacion\n\n" + "\n\n".join(_SENTENCIAS[:15]) + "\n\n"
    "### Tokens JWT\n\n" + "\n\n".join(_SENTENCIAS[15:30]) + "\n\n"
    "## Rate limiting\n\n" + "\n\n".join(_SENTENCIAS[30:45])
)

_DOCS_CORPUS = (
    sorted((DATA_DIR / "features").glob("*.md"))
    + sorted((DATA_DIR / "tutorial").glob("*.md"))
)


def test_chunking_rango_documento_real():
    """El doc real mas grande (features/dependencies.md) produce chunks 500-800."""
    texto = (DATA_DIR / "features" / "dependencies.md").read_text()
    chunks = build_chunks(texto, "features/dependencies.md")
    assert len(chunks) >= 1
    assert all(500 <= _tokens(c.page_content) <= 800 for c in chunks)


def test_chunking_rango_y_overlap_texto_largo():
    """Texto largo: todos los chunks en 500-800 con overlap ~100 entre consecutivos."""
    chunks = build_chunks(_TEXTO_LARGO, "features/guia.md")
    assert len(chunks) >= 2
    for c in chunks:
        assert 500 <= _tokens(c.page_content) <= 800
    for a, b in zip(chunks, chunks[1:]):
        assert 50 <= _solape_tokens(a.page_content, b.page_content) <= 150


def test_chunking_cabeceras_antepuestas():
    """Cada chunk arranca con la ruta h1-h3 de su seccion (D6)."""
    chunks = build_chunks(_TEXTO_SECCIONES, "features/guia.md")
    assert chunks[0].page_content.startswith("Guia de API > Autenticacion")
    assert "Guia de API > Rate limiting" in chunks[0].page_content
    for c in chunks:
        assert c.page_content.startswith("Guia de API")


def test_chunking_metadata_minima():
    """Metadata minima por chunk: source == document_id == nombre del .md."""
    chunks = build_chunks(_TEXTO_SECCIONES, "features/guia.md")
    for c in chunks:
        assert c.metadata["source"] == "guia.md"
        assert c.metadata["document_id"] == "guia.md"
        assert c.metadata["seccion"]  # ruta de cabeceras no vacia
        assert c.metadata["etiquetas"] == ["features"]


@pytest.mark.parametrize(
    "ruta",
    _DOCS_CORPUS,
    ids=[p.name for p in _DOCS_CORPUS],
)
def test_corpus_real_dentro_de_limite_superior(ruta):
    """Todo el corpus de U2 produce al menos un chunk de hasta 800 tokens."""
    texto = ruta.read_text()
    source = f"{ruta.parent.name}/{ruta.name}"
    chunks = build_chunks(texto, source)
    assert len(chunks) >= 1
    assert all(_tokens(c.page_content) <= 800 for c in chunks)
