# Dependencias con Depends

La inyección de dependencias en FastAPI se resuelve con `Depends`. Una
dependencia es una función que se ejecuta antes del endpoint y cuyo valor
de retorno se pasa como argumento. Sirve para autenticación, acceso a
base de datos, validaciones reutilizables y configuración compartida.

## Declarar una dependencia

Una dependencia es una función normal, a menudo sin parámetros. Para
usarla se declara el argumento del endpoint con `Depends`:

```python
from fastapi import Depends

def autenticar(token: str = Header(...)):
    return token

@app.get("/perfil")
def leer_perfil(token: str = Depends(autenticar)):
    return {"token": token}
```

## Compartir la misma dependencia

La misma dependencia se puede usar en varios endpoints. Por defecto
FastAPI cachea el resultado dentro de la misma petición: si dos
endpoints anidados dependen de la misma función, se ejecuta una sola vez.

## Sub-dependencias

Una dependencia puede depender de otra. FastAPI resuelve la cadena
completa en orden, de la más externa a la más interna:

```python
def verificar_base():
    return "conexion"

def get_db(base=Depends(verificar_base)):
    return base

@app.get("/items")
def listar_items(db=Depends(get_db)):
    return {"db": db}
```

## Dependencias con parámetros

Las dependencias pueden declarar query params, headers o cuerpos propios.
FastAPI los resuelve automáticamente, lo que permite construir helpers
como paginación reutilizable:

```python
def paginacion(limit: int = 10, skip: int = 0):
    return {"limit": limit, "skip": skip}

@app.get("/items")
def listar_items(page=Depends(paginacion)):
    return page
```

## Desactivar el caché con use_cache

Para ejecutar la dependencia en cada llamada (por ejemplo para obtener un
valor nuevo), usá `Depends(funcion, use_cache=False)`.

## Dependencias de clase

También se puede usar una clase con `__call__` como dependencia, lo que
permite combinar atributos de instancia con la lógica de resolución.
