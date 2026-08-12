# Path y query params en FastAPI

FastAPI valida los parámetros de ruta (path params) y los parámetros de
consulta (query params) a partir de los tipos que se declaran en la firma
de la función. No hace falta validación manual: el framework convierte,
valida y documenta cada parámetro automáticamente.

## Parámetros de ruta tipados

Los path params se extraen de la URL y se convierten al tipo declarado.
Si el cliente envía un valor que no se puede convertir, FastAPI responde
un error de validación 422:

```python
@app.get("/users/{user_id}")
def obtener_usuario(user_id: int):
    return {"user_id": user_id}
```

## Orden de parámetros con valores por defecto

En Python, un argumento sin valor por defecto no puede ir después de uno
que sí lo tiene. Por eso los path params (obligatorios) se declaran antes
que los query params (que suelen tener default).

## Parámetros de consulta (query params)

Los query params se leen de la URL después del signo `?` (por ejemplo
`/items?limit=10&skip=0`). Se declaran como argumentos con valor por
defecto y ese default se usa cuando el cliente no los envía:

```python
@app.get("/items")
def listar_items(limit: int = 10, skip: int = 0):
    return {"limit": limit, "skip": skip}
```

### Parámetros opcionales

Un query param es opcional si su tipo admite `None`:

```python
def buscar(q: str | None = None):
    ...
```

### Parámetros obligatorios

Un query param sin valor por defecto es obligatorio: si el cliente no lo
envía, FastAPI responde 422 con el detalle del campo faltante.

## Restricciones con Query y Path

Las restricciones de validación se declaran con `Query` y `Path`, que
aceptan valores mínimos, máximos y patrones:

```python
from fastapi import Query, Path

@app.get("/items/{item_id}")
def leer_item(
    item_id: int = Path(ge=1),
    limit: int = Query(default=10, ge=1, le=100),
    q: str | None = Query(default=None, max_length=50),
):
    return {"item_id": item_id, "limit": limit, "q": q}
```

Si el cliente envía `limit=500`, la respuesta es 422 porque excede el
máximo declarado. Estas restricciones aparecen también en la
documentación OpenAPI.

## Conversión de tipos

FastAPI convierte strings de la URL al tipo declarado y documenta la
conversión en el esquema:

| Tipo | Acepta en la URL |
|---|---|
| `int` | enteros (`42`, `-1`) |
| `float` | decimales (`3.14`) |
| `bool` | `true`, `false`, `1`, `0`, `on`, `off` |
| `datetime` | fechas ISO 8601 (`2024-01-01T10:00:00Z`) |
| `Enum` | solo los valores definidos en la enumeración |

La conversión fallida produce un 422 con el detalle del parámetro.

## Parámetros de cabecera y cookie

Además de path y query, FastAPI lee cabeceras y cookies con los mismos
mecanismos de tipos:

```python
from fastapi import Header

@app.get("/items")
def listar_items(user_agent: str | None = Header(default=None)):
    return {"user_agent": user_agent}
```

Los parámetros de cabecera se documentan en OpenAPI y permiten mover la
configuración de la petición (tokens, ids de correlación, versiones) fuera
de la URL sin perder la validación automática.
