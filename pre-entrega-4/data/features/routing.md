# Routing en FastAPI

Las rutas (o endpoints) son la base de toda API: asocian una URL con una
función que responde a una petición HTTP. En FastAPI se definen con el
decorador del método HTTP sobre una función asíncrona o síncrona. La
organización de las rutas determina la mantenibilidad del proyecto a
medida que crece el número de recursos.

## Decoradores de métodos HTTP

Cada método HTTP tiene su propio decorador. El primer argumento es la ruta
y se puede pasar un parámetro `status_code` y `tags` para agrupar:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items")
def listar_items():
    return {"items": []}

@app.post("/items", status_code=201)
def crear_item():
    return {"ok": True}
```

| Decorador | Método HTTP | Uso típico |
|---|---|---|
| `@app.get` | GET | Consultar recursos (sin efectos secundarios) |
| `@app.post` | POST | Crear recursos |
| `@app.put` | PUT | Reemplazar un recurso completo |
| `@app.patch` | PATCH | Actualización parcial |
| `@app.delete` | DELETE | Eliminar recursos |
| `@app.options` / `@app.head` | OPTIONS / HEAD | Metadatos y cabeceras sin cuerpo |

## Orden de definición

FastAPI evalúa las rutas en el orden en que se definen. Si una ruta
dinámica (con parámetro) aparece antes que una fija, puede capturar
peticiones que deberían ir a la fija. Este ejemplo es un error clásico:

```python
@app.get("/users/{user_id}")
def usuario(user_id: int):
    ...

@app.get("/users/me")
def usuario_actual():
    ...
```

Toda petición a `/users/me` cae en la primera ruta e intenta convertir
"me" a `int`, devolviendo 422. La regla es definir primero las rutas más
específicas.

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

## Respuestas y códigos de estado

El valor que devuelve la función se serializa automáticamente a JSON. Para
códigos de estado distintos de 200, se usa el parámetro `status_code` del
decorador o `JSONResponse` cuando la respuesta necesita cabeceras propias:

```python
from fastapi.responses import JSONResponse

@app.get("/items/{item_id}")
def leer_item(item_id: int):
    if item_id == 0:
        return JSONResponse(status_code=404, content={"error": "no existe"})
    return {"item_id": item_id}
```

## Organización con APIRouter

Para separar el código por dominio, las rutas se agrupan en `APIRouter` y
se registran en la aplicación con `include_router`. Esto permite modularizar
recursos grandes y reutilizar prefijos y dependencias comunes:

```python
from fastapi import APIRouter

router = APIRouter(prefix="/items", tags=["items"])

@router.get("")
def listar_items():
    return {"items": []}

app.include_router(router)
```

Con `prefix="/items"`, el endpoint queda en `GET /items`. Los `tags`
agrupan las rutas en la documentación interactiva y facilitan la
navegación cuando el proyecto tiene decenas de endpoints.
