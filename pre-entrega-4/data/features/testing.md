# Testing de APIs con TestClient

FastAPI trae integración directa con `TestClient` de Starlette, que permite
probar los endpoints sin levantar un servidor. Se usa con pytest y se
puede combinar con la inyección de dependencias para aislar el sistema.

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

## Probar errores de validación

Los tests también verifican los errores esperados: un 422 por datos
inválidos, un 401 por falta de autenticación o un 404 por una ruta
inexistente.

## Reemplazar dependencias

Con `app.dependency_overrides` se sustituye una dependencia real por un
doble de prueba. El override se aplica con dict y se limpia con `clear`
al final del test:

```python
app.dependency_overrides[autenticar] = lambda: "usuario-de-test"
```

Esto permite probar endpoints protegidos sin credenciales reales.
