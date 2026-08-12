# Primeros pasos: la primera API

Este paso crea la primera aplicación FastAPI, la ejecuta con Uvicorn y
explora la documentación interactiva que genera el framework.

## Crear la aplicación

Se define una instancia de `FastAPI` y un primer endpoint con `@app.get`:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def raiz():
    return {"mensaje": "Hola mundo"}
```

## Ejecutar el servidor

Con Uvicorn se levanta el servidor indicando el módulo y la variable de
la aplicación. La bandera `--reload` reinicia el servidor ante cambios:

```bash
uvicorn main:app --reload
```

La API queda disponible en `http://127.0.0.1:8000`.

## Documentación interactiva

FastAPI genera automáticamente documentación OpenAPI. La interfaz Swagger
permite probar los endpoints desde el navegador:

- `http://127.0.0.1:8000/docs` — interfaz interactiva de Swagger UI.
- `http://127.0.0.1:8000/redoc` — documentación alternativa con ReDoc.
- `http://127.0.0.1:8000/openapi.json` — el esquema OpenAPI en JSON.

## Probar el endpoint

Desde el navegador o con `curl`:

```bash
curl http://127.0.0.1:8000/
```

La respuesta es el JSON `{"mensaje": "Hola mundo"}`. El esquema en
`/docs` ya muestra el endpoint, su método y el tipo de respuesta.
