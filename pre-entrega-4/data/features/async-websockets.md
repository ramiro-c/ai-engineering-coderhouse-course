# Async y WebSockets en FastAPI

FastAPI está construido sobre Starlette y soporta tanto endpoints
síncronos (`def`) como asíncronos (`async def`), además de conexiones
WebSocket para comunicación bidireccional en tiempo real. Elegir el tipo
correcto de función determina el rendimiento bajo concurrencia.

## Endpoints async

Los endpoints con `async def` corren en el event loop y pueden usar
`await` para operaciones de entrada y salida (llamadas HTTP, lecturas de
base de datos):

```python
@app.get("/datos")
async def leer_datos():
    resultado = await servicio_externo.obtener()
    return resultado
```

## def vs async def

| Tipo de función | Cómo la ejecuta FastAPI | Uso recomendado |
|---|---|---|
| `def` | Thread pool separado | Trabajo de CPU, librerías síncronas (SQLAlchemy clásico, requests) |
| `async def` | Event loop directo | I/O asíncrona: `httpx.AsyncClient`, drivers async, WebSockets |

Si la función del endpoint es síncrona (`def`), FastAPI la ejecuta en un
thread pool para no bloquear el event loop. Se usa `async def` solo
cuando la operación sea realmente asíncrona: un `def` con trabajo de CPU
evita el costo de un event loop bloqueado, y un `async def` con código
bloqueante congela todas las peticiones concurrentes.

## WebSockets

Un WebSocket permite enviar y recibir mensajes en ambas direcciones. La
conexión se acepta con `accept`, se reciben mensajes con `receive_text`
y se envían con `send_text`:

```python
from fastapi import WebSocket

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        mensaje = await websocket.receive_text()
        await websocket.send_text(f"eco: {mensaje}")
```

## Manejar desconexiones

Si el cliente cierra la conexión, se lanza `WebSocketDisconnect`. Se
captura para limpiar recursos o remover el cliente de una lista de
conexiones activas:

```python
from fastapi import WebSocketDisconnect

try:
    while True:
        await websocket.receive_text()
except WebSocketDisconnect:
    conexiones.remove(websocket)
```

## Múltiples conexiones

Para un chat o notificaciones, se mantiene una lista de WebSockets
activos y se difunde el mensaje a todos los conectados. Cada conexión
debe registrarse al aceptar y eliminarse al desconectarse:

```python
conexiones: list[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    conexiones.append(websocket)
    try:
        while True:
            mensaje = await websocket.receive_text()
            for cliente in list(conexiones):
                await cliente.send_text(f"{mensaje}")
    except WebSocketDisconnect:
        conexiones.remove(websocket)
```

La lista se copia con `list(conexiones)` al iterar porque un cliente
puede desconectarse durante el envío y modificar la colección. La
difusión es secuencial: si la cantidad de conexiones crece, el envío
masivo se delega a una tarea en background para no bloquear el receive
del bucle.
