# Primeros pasos: la primera API

Este paso crea la primera aplicación FastAPI, la ejecuta con Uvicorn y
explora la documentación interactiva que genera el framework. Al final
de este recorrido queda un endpoint funcional validado con la propia
interfaz de Swagger UI.

## Crear la aplicación

Se define una instancia de `FastAPI` y un primer endpoint con `@app.get`:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def raiz():
    return {"mensaje": "Hola mundo"}
```

El decorador asocia la URL `/` con el método HTTP `GET` y la función
`raiz`. El valor de retorno, un diccionario, se serializa automáticamente
a JSON en la respuesta. No hace falta declarar el tipo de contenido:
FastAPI lo deriva del valor devuelto.

## Ejecutar el servidor

Con Uvicorn se levanta el servidor indicando el módulo y la variable de
la aplicación. La bandera `--reload` reinicia el servidor ante cambios
en el código, útil durante el desarrollo:

```bash
uvicorn main:app --reload
```

La API queda disponible en `http://127.0.0.1:8000`. El argumento
`main:app` se lee como "módulo `main`, variable `app`": el archivo se
llama `main.py` y la instancia de FastAPI se llama `app`.

## Documentación interactiva

FastAPI genera automáticamente documentación OpenAPI a partir de los
tipos declarados en el código. La interfaz Swagger permite probar los
endpoints desde el navegador sin herramientas externas:

- `http://127.0.0.1:8000/docs` — interfaz interactiva de Swagger UI.
- `http://127.0.0.1:8000/redoc` — documentación alternativa con ReDoc.
- `http://127.0.0.1:8000/openapi.json` — el esquema OpenAPI en JSON.

## Probar el endpoint

Desde el navegador o con `curl`:

```bash
curl http://127.0.0.1:8000/
```

La respuesta es el JSON `{"mensaje": "Hola mundo"}`. El esquema en
`/docs` ya muestra el endpoint, su método y el tipo de respuesta. Desde
esa misma interfaz se puede ejecutar el `GET /` y ver el resultado sin
escribir código adicional.

## El esquema OpenAPI generado

El archivo `openapi.json` es la fuente que usan Swagger UI y ReDoc. Su
estructura incluye el título de la aplicación, las rutas registradas y
los esquemas de los modelos de Pydantic. Si el proyecto se integra con
clientes generados (OpenAPI Generator, fetch en el frontend), este
esquema es el contrato que define la comunicación.

## Siguientes pasos

Con el servidor y la documentación funcionando, el siguiente paso del
tutorial es construir una API completa: modelo de datos con Pydantic,
endpoints de creación y consulta, y almacenamiento en memoria.
