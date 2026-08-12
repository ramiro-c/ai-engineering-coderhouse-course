# Middleware en FastAPI

Un middleware es código que se ejecuta antes y después de cada petición,
envolviendo el manejo de la ruta. Se usa para logging, medición de
tiempos, cabeceras personalizadas, compresión y control de acceso. Al
estar por encima del enrutado, aplica a toda la aplicación sin tocar cada
endpoint.

## Middleware HTTP con el decorador

El decorador `@app.middleware("http")` recibe la petición y una función
`call_next` que ejecuta el resto del procesamiento. El código previo a
`call_next` corre antes del endpoint y el posterior, después:

```python
from fastapi import FastAPI, Request

app = FastAPI()

@app.middleware("http")
async def agregar_cabecera(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Proceso"] = "ok"
    return response
```

## Medición de tiempos de respuesta

Un middleware típico mide la duración de cada petición con `perf_counter`
y la expone en una cabecera o en el log:

```python
import time

@app.middleware("http")
async def medir_tiempo(request: Request, call_next):
    inicio = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Duracion-ms"] = str(
        round((time.perf_counter() - inicio) * 1000, 2)
    )
    return response
```

## BaseHTTPMiddleware

La clase `BaseHTTPMiddleware` de Starlette permite definir el mismo
comportamiento con `dispatch` y se agrega con `app.add_middleware`:

```python
from starlette.middleware.base import BaseHTTPMiddleware

class MiMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        return response

app.add_middleware(MiMiddleware)
```

## Middlewares incluidos en Starlette

| Middleware | Función |
|---|---|
| `CORSMiddleware` | Permite peticiones desde otros orígenes |
| `GZipMiddleware` | Comprime respuestas mayores a un umbral |
| `TrustedHostMiddleware` | Rechaza peticiones con Host no permitido |
| `HTTPSRedirectMiddleware` | Redirige HTTP a HTTPS |
| `SessionMiddleware` | Firma una cookie de sesión por petición |

## CORS

Para permitir peticiones desde un frontend en otro origen se usa el
middleware CORS de Starlette, configurado con la lista de orígenes
permitidos:

```python
from starlette.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

En producción, `allow_origins` debe listar los dominios reales del
frontend y no usar `*` cuando las peticiones incluyen credenciales.

## Orden de ejecución

Los middlewares se ejecutan en el orden inverso al que se agregan: el
último agregado es el primero en recibir la petición. El orden importa
cuando varios middlewares modifican la petición o la respuesta: por
ejemplo, el que agrega cabeceras de seguridad debe correr después del
que comprime, para no comprimir cabeceras ya escritas. El mismo patrón
aplica a las excepciones: un middleware que registra errores debe
envolver a los demás para capturar todas las fallas.
