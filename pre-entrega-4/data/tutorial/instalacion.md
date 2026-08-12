# Instalación de FastAPI

Antes de escribir la primera API hay que preparar el entorno: crear un
entorno virtual e instalar FastAPI con el servidor Uvicorn.

## Requisitos

FastAPI requiere Python 3.8 o superior. Se recomienda usar un entorno
virtual para aislar las dependencias del proyecto.

## Crear el entorno virtual

Desde la carpeta del proyecto se crea el entorno con `venv` y se activa:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

En Windows la activación es `.venv\Scripts\activate`.

## Instalar FastAPI y Uvicorn

FastAPI es el framework y Uvicorn el servidor ASGI que sirve la
aplicación. La instalación con `pip` incluye ambas piezas:

```bash
pip install "fastapi[standard]"
```

La variante `[standard]` instala extras útiles de desarrollo. Para una
instalación mínima alcanza con `pip install fastapi uvicorn`.

## Verificar la instalación

Se comprueba que los paquetes quedaron instalados y sus versiones:

```bash
python -c "import fastapi, uvicorn; print(fastapi.__version__, uvicorn.__version__)"
```

## Primer chequeo rápido

Con la instalación lista, se puede crear un archivo `main.py` mínimo con
un endpoint y ejecutarlo (paso siguiente en el tutorial). Si aparece el
mensaje de Uvicorn con la URL local, el entorno quedó correctamente
configurado.
