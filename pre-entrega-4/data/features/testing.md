# Testing de APIs con TestClient

FastAPI trae integración directa con `TestClient` de Starlette, que permite
probar los endpoints sin levantar un servidor. Se usa con pytest y se
puede combinar con la inyección de dependencias para aislar el sistema.
Los tests se convierten en la red de seguridad que permite refactorizar
sin romper el contrato de la API.

## Crear el cliente de test

`TestClient` envuelve la aplicación y se usa como contexto para que los
eventos de startup se ejecuten:

```python
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_leer_items():
    response = client.get("/items")
    assert response.status_code == 200
    assert response.json() == {"items": []}
```

## TestClient como fixture

Para reutilizar el cliente en varios tests se define como fixture de
pytest, con alcance de módulo o de sesión:

```python
import pytest
from fastapi.testclient import TestClient

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
```

## Probar peticiones con cuerpo y cabeceras

`client.post` acepta `json` para el cuerpo, `headers` para cabeceras
personalizadas y `params` para query params:

```python
def test_crear_item(client):
    response = client.post("/items", json={"nombre": "teclado"})
    assert response.status_code == 201
    assert response.json()["nombre"] == "teclado"
```

## Verificaciones habituales

| Verificación | Ejemplo |
|---|---|
| Código de estado | `assert response.status_code == 201` |
| Cuerpo exacto | `assert response.json() == {"items": []}` |
| Campo presente | `assert "id" in response.json()` |
| Cabecera | `assert response.headers["X-Proceso"] == "ok"` |
| Error de validación | `assert response.status_code == 422` |

## Probar errores de validación

Los tests también verifican los errores esperados: un 422 por datos
inválidos, un 401 por falta de autenticación o un 404 por una ruta
inexistente. Validar el camino de error es tan importante como el
camino feliz, porque documenta el comportamiento real del contrato:

```python
def test_validacion_item(client):
    response = client.post("/items", json={})
    assert response.status_code == 422
    assert "nombre" in response.text
```

## Reemplazar dependencias

Con `app.dependency_overrides` se sustituye una dependencia real por un
doble de prueba. El override se aplica con dict y se limpia con `clear`
al final del test:

```python
app.dependency_overrides[autenticar] = lambda: "usuario-de-test"
```

Esto permite probar endpoints protegidos sin credenciales reales y
sustituir la base de datos por un repositorio en memoria.

## Testing de WebSockets

`TestClient` también cubre conexiones WebSocket con `websocket_connect`:

```python
def test_websocket(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text("hola")
        assert ws.receive_text() == "eco: hola"
```

La suite de tests del proyecto debe correr sin red ni credenciales: las
llamadas a servicios externos se reemplazan por dobles y los tests de
integración que sí requieren infraestructura se marcan como `slow` para
ejecutarlos aparte.
