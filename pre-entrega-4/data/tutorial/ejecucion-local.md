# Ejecución local de la aplicación

Este paso cubre cómo ejecutar la aplicación en desarrollo: opciones de
Uvicorn, variables de entorno y resolución de errores de arranque. Un
flujo de ejecución claro acelera el ciclo de desarrollo y evita
sorpresas al pasar de la máquina local a un entorno de despliegue.

## Comando básico de Uvicorn

El servidor se ejecuta con `uvicorn` indicando módulo y variable de la
aplicación:

```bash
uvicorn main:app
```

## Opciones útiles de ejecución

| Opción | Efecto |
|---|---|
| `--reload` | Reinicia el servidor al guardar cambios; ideal para desarrollo |
| `--port 8000` | Cambia el puerto (por defecto 8000) |
| `--host 0.0.0.0` | Expone el servidor a la red local; útil para probar desde el celular o contenedores |
| `--workers 4` | Levanta varios procesos (producción; no compatible con `--reload`) |
| `--log-level info` | Controla la verbosidad del log (debug, info, warning, error) |

```bash
uvicorn main:app --reload --port 8000
```

## Configuración con variables de entorno

Los valores sensibles o de entorno se leen desde un archivo `.env` con
`python-dotenv` y se acceden con `os.getenv`. El archivo `.env` no se
versiona; el repositorio incluye un `.env.example` con las claves
esperadas:

```python
from dotenv import load_dotenv
import os

load_dotenv()
puerto = int(os.getenv("PUERTO", "8000"))
```

El segundo argumento de `getenv` es el valor por defecto: así la
aplicación arranca sin configuración previa y los secretos (claves de
API, credenciales de base de datos) quedan fuera del control de
versiones.

## Errores comunes de arranque

- `ModuleNotFoundError`: el módulo indicado no existe o falta instalar la
  dependencia. Verificar el nombre del archivo y el `pip install`.
- `ImportError: cannot import name 'app'`: la variable `app` no existe en
  el módulo. Revisar que la instancia `FastAPI()` se llame igual que la
  indicada en el comando.
- `Address already in use`: el puerto está ocupado. Cambiar el puerto con
  `--port` o liberar el proceso que lo usa (`lsof -i :8000` en macOS y
  Linux).

## Detener el servidor

Se detiene con `Ctrl+C`. El log de Uvicorn muestra el inicio, las
peticiones recibidas y cualquier error en tiempo de ejecución, lo que
ayuda a depurar la API durante el desarrollo. Cada reinicio con
`--reload` vuelve a emitir el banner de arranque con la URL y los
endpoints documentados.
