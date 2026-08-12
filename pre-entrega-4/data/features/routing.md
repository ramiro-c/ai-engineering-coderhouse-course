# Routing en FastAPI

Las rutas (o endpoints) son la base de toda API: asocian una URL con una
función que responde a una petición HTTP. En FastAPI se definen con el
decorador del método HTTP sobre una función asíncrona o síncrona.

## Decoradores de métodos HTTP

Cada método HTTP tiene su propio decorador. El primer argumento es la ruta
y se puede pasar un parámetro `status_code` y `tags` para agrupar:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items")
def listar_items():
    return {"items": []}

@app.post("/items")
def crear_item():
    return {"ok": True}
```

Los decoradores disponibles son `@app.get`, `@app.post`, `@app.put`,
`@app.delete`, `@app.patch`, `@app.options` y `@app.head`.

## Orden de definición

FastAPI evalúa las rutas en el orden en que se definen. Si una ruta
dinámica (con parámetro) aparece antes que una fija, puede capturar
peticiones que deberían ir a la fija. Definí primero las rutas más
específicas.

## Respuestas y códigos de estado

El valor que devuelve la función se serializa automáticamente a JSON. Para
códigos de estado distintos de 200, usá el parámetro `status_code` del
decorador o `JSONResponse`.

## Rutas dinámicas

Los parámetros de ruta se escriben entre llaves en la URL y se reciben como
argumentos de la función:

```python
@app.get("/items/{item_id}")
def leer_item(item_id: int):
    return {"item_id": item_id}
```

Si el tipo declarado no coincide con el valor de la URL, FastAPI devuelve
un error de validación 422 en lugar de fallar en tiempo de ejecución.
