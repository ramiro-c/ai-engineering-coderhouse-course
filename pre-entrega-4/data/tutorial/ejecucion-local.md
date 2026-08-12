# Ejecución local de la aplicación

Este paso cubre cómo ejecutar la aplicación en desarrollo: opciones de
Uvicorn, variables de entorno y resolución de errores de arranque.

## Comando básico de Uvicorn

El servidor se ejecuta con `uvicorn` indicando módulo y variable de la
aplicación:

```bash
uvicorn main:app
```

## Opciones útiles de ejecución

- `--reload`: reinicia el servidor automáticamente al guardar cambios,
  ideal para desarrollo.
- `--port 8000`: cambia el puerto (por defecto 8000).
- `--host 0.0.0.0`: expone el servidor a la red local, útil para probar
  desde el celular o contenedores.

```bash
uvicorn main:app --reload --port 8000
```

## Configuración con variables de entorno

Los valores sensibles o de entorno se leen desde un archivo `.env` con
`python-dotenv` y se acceden con `os.getenv`. El archivo `.env` no se
versiona; el repositorio incluye un `.env.example` con las claves
esperadas.

## Errores comunes de arranque

- `ModuleNotFoundError`: el módulo indicado no existe o falta instalar la
  dependencia. Verificá el nombre del archivo y el `pip install`.
- `ImportError: cannot import name 'app'`: la variable `app` no existe en
  el módulo. Revisá que la instancia `FastAPI()` se llame igual que la
  indicada en el comando.
- `Address already in use`: el puerto está ocupado. Cambiá el puerto con
  `--port` o liberá el proceso que lo usa.

## Detener el servidor

Se detiene con `Ctrl+C`. El log de Uvicorn muestra el inicio, las
peticiones recibidas y cualquier error en tiempo de ejecución, lo que
ayuda a depurar la API durante el desarrollo.
