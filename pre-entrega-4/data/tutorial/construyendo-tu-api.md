# Construyendo tu API paso a paso

Este paso arma una API funcional completa: un modelo de datos, endpoints
de creación y consulta, y un almacenamiento en memoria.

## Paso 1: definir el modelo

Se declara el modelo de datos con Pydantic. Los campos tipados definen
la validación automática del cuerpo de la petición:

```python
from pydantic import BaseModel

class Tarea(BaseModel):
    titulo: str
    completada: bool = False
```

## Paso 2: crear el endpoint de alta

El endpoint `POST` recibe una `Tarea`, la guarda en un diccionario en
memoria y devuelve la tarea creada con su identificador:

```python
from fastapi import FastAPI

app = FastAPI()
tareas = {}
contador = 0

@app.post("/tareas")
def crear_tarea(tarea: Tarea):
    global contador
    contador += 1
    tareas[contador] = tarea
    return {"id": contador, **tarea.model_dump()}
```

## Paso 3: consultar con path params

El endpoint `GET` con path param devuelve la tarea por id o un error 404
si no existe:

```python
from fastapi import HTTPException

@app.get("/tareas/{tarea_id}")
def leer_tarea(tarea_id: int):
    if tarea_id not in tareas:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    return tareas[tarea_id]
```

## Paso 4: listar con query params

El endpoint de listado usa query params para paginar la respuesta:

```python
@app.get("/tareas")
def listar_tareas(limit: int = 10, skip: int = 0):
    return list(tareas.values())[skip : skip + limit]
```

## Paso 5: probar en /docs

Con el servidor corriendo, la documentación en `/docs` permite probar los
cuatro endpoints: crear, leer, listar y ver los errores de validación al
enviar datos inválidos.
