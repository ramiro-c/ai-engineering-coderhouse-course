# Instalación de FastAPI

Antes de escribir la primera API hay que preparar el entorno: crear un
entorno virtual e instalar FastAPI con el servidor Uvicorn. Este paso
garantiza que las dependencias del proyecto queden aisladas del resto
del sistema y que la versión instalada sea reproducible.

## Requisitos

FastAPI requiere Python 3.8 o superior. Se recomienda usar un entorno
virtual para aislar las dependencias del proyecto y evitar conflictos
entre versiones de distintas aplicaciones.

| Requisito | Valor recomendado |
|---|---|
| Python | 3.8 o superior (3.10+ para las anotaciones `X \| None`) |
| Gestor de paquetes | `pip` incluido en la instalación de Python |
| Sistema operativo | Linux, macOS o Windows (WSL recomendado) |

## Crear el entorno virtual

Desde la carpeta del proyecto se crea el entorno con `venv` y se activa:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

En Windows la activación es `.venv\Scripts\activate`. Una vez activo,
el prompt del terminal muestra el nombre del entorno entre paréntesis,
lo que confirma que los comandos `pip` apuntan a ese entorno y no al
Python global del sistema.

## Instalar FastAPI y Uvicorn

FastAPI es el framework y Uvicorn el servidor ASGI que sirve la
aplicación. La instalación con `pip` incluye ambas piezas:

```bash
pip install "fastapi[standard]"
```

La variante `[standard]` instala extras útiles de desarrollo: Uvicorn,
`httptools` para mayor rendimiento y `python-multipart` para formularios
y subida de archivos. Para una instalación mínima alcanza con:

```bash
pip install fastapi uvicorn
```

Si el proyecto usará plantillas o autenticación OAuth2, conviene agregar
`jinja2` y `python-multipart` explícitamente. Toda dependencia extra se
debe registrar en `requirements.txt` para que el entorno sea
reproducible en otra máquina o en un contenedor.

## Verificar la instalación

Se comprueba que los paquetes quedaron instalados y sus versiones:

```bash
python -c "import fastapi, uvicorn; print(fastapi.__version__, uvicorn.__version__)"
```

Un error `ModuleNotFoundError` indica que el paquete no se instaló en el
entorno activo, por ejemplo porque se omitió la activación del `venv` o
porque el comando `pip` apunta a otro intérprete.

## Errores comunes de instalación

- `pip` no encontrado: usar `python -m pip` en lugar de `pip` directo.
- Permisos denegados: instalar siempre dentro del entorno virtual; si el
  error persiste, revisar que el `venv` esté activo.
- Versión de Python antigua: actualizar a 3.10 o superior para usar las
  anotaciones de tipos modernas que FastAPI documenta en sus ejemplos.

## Primer chequeo rápido

Con la instalación lista, se puede crear un archivo `main.py` mínimo con
un endpoint y ejecutarlo (paso siguiente en el tutorial). Si aparece el
mensaje de Uvicorn con la URL local, el entorno quedó correctamente
configurado y se puede continuar con los primeros pasos.
