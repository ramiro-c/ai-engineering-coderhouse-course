# Construyendo tu API paso a paso

Este paso arma una API funcional completa: un modelo de datos, endpoints
de creación, consulta y actualización, y un almacenamiento en memoria.
Es la base mínima de un CRUD y permite ver en acción la validación
automática de FastAPI.

## Paso 1: definir el modelo

Se declara el modelo de datos con Pydantic. Los campos tipados definen
la validación automática del cuerpo de la petición:

```python
from pydantic import BaseModel

class Tarea(BaseModel):
    titulo: str
    completada: bool = False
```

El campo `titulo` es obligatorio; si el cliente no lo envía, FastAPI
responde un error 422 antes de ejecutar la función. `completada` tiene
un valor por defecto y por eso es opcional en la entrada.

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

`model_dump()` convierte la instancia de Pydantic en un diccionario. El
código de estado por defecto de un `POST` exitoso es 200; para indicar
que se creó un recurso nuevo se puede declarar `status_code=201` en el
decorador.

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

El path param `tarea_id` se declara con tipo `int`: si el cliente envía
un valor no numérico, FastAPI responde 422 con el detalle de la
validación fallida.

## Paso 4: listar con query params

El endpoint de listado usa query params para paginar la respuesta:

```python
@app.get("/tareas")
def listar_tareas(limit: int = 10, skip: int = 0):
    return list(tareas.values())[skip : skip + limit]
```

Los query params `limit` y `skip` son opcionales porque tienen valor por
defecto. La URL `/tareas?limit=5&skip=10` devuelve las tareas de la
posición 10 en adelante, cinco por página.

## Paso 5: actualizar una tarea

Un endpoint `PUT` permite reemplazar o modificar el estado de una tarea
existente:

```python
@app.put("/tareas/{tarea_id}")
def actualizar_tarea(tarea_id: int, tarea: Tarea):
    if tarea_id not in tareas:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    tareas[tarea_id] = tarea
    return {"id": tarea_id, **tarea.model_dump()}
```

Combinar el path param con el modelo en la firma muestra cómo FastAPI
resuelve cada argumento por su tipo: el id viene de la URL y el cuerpo
se valida contra el modelo.

## Paso 6: probar en /docs

Con el servidor corriendo, la documentación en `/docs` permite probar
los cinco endpoints: crear, leer, listar, actualizar y ver los errores
de validación al enviar datos inválidos. El almacenamiento es en memoria:
al reiniciar el servidor, los datos se pierden. Para persistencia real se
reemplaza el diccionario por una base de datos dentro de una dependencia.
