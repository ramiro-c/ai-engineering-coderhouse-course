# Seguridad en FastAPI

FastAPI integra los mecanismos de seguridad de OpenAPI: HTTP Basic,
OAuth2 con password flow, bearer tokens y JWT. La autenticación se
combina con las dependencias para proteger endpoints de forma declarativa.

## HTTP Basic

Se usa la clase `HTTPBasic` de Starlette y la credencial se valida
comparando el usuario y la contraseña con `secrets.compare_digest`:

```python
from fastapi.security import HTTPBasic, HTTPBasicCredentials

seguridad = HTTPBasic()

@app.get("/privado")
def privado(credenciales: HTTPBasicCredentials = Depends(seguridad)):
    return {"usuario": credenciales.username}
```

## OAuth2 con password flow

`OAuth2PasswordBearer` indica que la API espera un token en el header
`Authorization` con el esquema `Bearer`. El token se valida en una
dependencia:

```python
from fastapi.security import OAuth2PasswordBearer

oauth2 = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/perfil")
def perfil(token: str = Depends(oauth2)):
    return {"token": token}
```

## OAuth2PasswordRequestForm

El formulario de login se recibe con `OAuth2PasswordRequestForm`, que
espera los campos `username` y `password` en formato de formulario (no
JSON). Se usa para emitir el token:

```python
from fastapi.security import OAuth2PasswordRequestForm

@app.post("/token")
def login(form: OAuth2PasswordRequestForm = Depends()):
    return {"access_token": form.username, "token_type": "bearer"}
```

## JWT

Los tokens JWT se firman con una librería como PyJWT. El payload contiene
el subject (`sub`) y la expiración (`exp`), y la firma usa un secreto. La
dependencia de autenticación decodifica el token y devuelve el usuario.

## Scopes

Los scopes son permisos declarados en el token OAuth2. `OAuth2SecurityScopes`
permite validar en el endpoint qué scopes debe tener el token para
autorizar la operación.
