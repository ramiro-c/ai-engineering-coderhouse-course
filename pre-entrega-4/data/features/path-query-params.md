# Path y query params en FastAPI

FastAPI valida los parámetros de ruta (path params) y los parámetros de
consulta (query params) a partir de los tipos que declarás en la firma de
la función. No hace falta validación manual.

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

## Conversión de tipos

FastAPI convierte strings de la URL al tipo declarado: `int`, `float`,
`bool` (acepta `true`, `false`, `1`, `0`) y `datetime`, entre otros. La
conversión fallida produce un 422 con el detalle del parámetro.
