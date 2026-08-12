# Dependencias con Depends

La inyección de dependencias en FastAPI se resuelve con `Depends`. Una
dependencia es una función que se ejecuta antes del endpoint y cuyo valor
de retorno se pasa como argumento. Sirve para autenticación, acceso a
base de datos, validaciones reutilizables y configuración compartida, sin
acoplar el endpoint a la infraestructura concreta.

## Declarar una dependencia

Una dependencia es una función normal, a menudo sin parámetros. Para
usarla se declara el argumento del endpoint con `Depends`:

```python
from fastapi import Depends, Header

def autenticar(token: str = Header(...)):
    return token

@app.get("/perfil")
def leer_perfil(token: str = Depends(autenticar)):
    return {"token": token}
```

## Casos de uso habituales

| Caso | Qué devuelve la dependencia |
|---|---|
| Autenticación | Usuario o token validado |
| Base de datos | Sesión o conexión (con `yield` para el cierre) |
| Paginación | Objeto con `limit` y `skip` ya validados |
| Configuración | Settings cargados desde el entorno |
| Autorización | Permisos del usuario para el recurso |

## Compartir la misma dependencia

La misma dependencia se puede usar en varios endpoints. Por defecto
FastAPI cachea el resultado dentro de la misma petición: si dos
endpoints anidados dependen de la misma función, se ejecuta una sola vez.
Esto evita consultas repetidas a base de datos dentro de una misma
petición.

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

La composición de sub-dependencias permite construir capas: autenticar,
luego abrir la sesión, luego cargar el usuario, y entregar al endpoint
solo lo que necesita.

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

## Dependencias con yield

Para recursos que requieren limpieza (sesiones, conexiones, archivos), la
dependencia se declara como generador: el código posterior al `yield` se
ejecuta al terminar la petición, incluso si el endpoint lanza una
excepción:

```python
def get_db():
    db = conectar()
    try:
        yield db
    finally:
        db.close()
```

## Desactivar el caché con use_cache

Para ejecutar la dependencia en cada llamada (por ejemplo para obtener un
valor nuevo), se usa `Depends(funcion, use_cache=False)`. Es el caso de
timestamps o lecturas que deben reflejar el estado actual.

## Dependencias de clase

También se puede usar una clase con `__call__` como dependencia, lo que
permite combinar atributos de instancia con la lógica de resolución y
reutilizar configuraciones complejas sin funciones de fábrica.
