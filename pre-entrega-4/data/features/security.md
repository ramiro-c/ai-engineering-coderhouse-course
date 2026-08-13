# Seguridad en FastAPI

FastAPI integra los mecanismos de seguridad de OpenAPI: HTTP Basic,
OAuth2 con password flow, bearer tokens y JWT. La autenticación se
combina con las dependencias para proteger endpoints de forma declarativa,
y cada esquema queda documentado automáticamente en Swagger UI.

## HTTP Basic

Se usa la clase `HTTPBasic` de Starlette y la credencial se valida
comparando el usuario y la contraseña con `secrets.compare_digest`, que
resiste ataques de timing:

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

`tokenUrl` señala el endpoint que emite el token, de modo que la
documentación interactiva ofrece el botón "Authorize" para ingresar las
credenciales y probar los endpoints protegidos.

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
dependencia de autenticación decodifica el token y devuelve el usuario:

```python
import jwt

SECRETO = "clave-secreta-de-prueba"

def crear_token(usuario: str):
    payload = {"sub": usuario, "exp": datetime.utcnow() + timedelta(minutes=30)}
    return jwt.encode(payload, SECRETO, algorithm="HS256")

def validar_token(token: str = Depends(oauth2)):
    return jwt.decode(token, SECRETO, algorithms=["HS256"])
```

El secreto se lee de una variable de entorno, nunca se hardcodea. La
expiración limita la ventana de validez del token: al vencer, el cliente
debe autenticarse de nuevo o usar un refresh token.

## Hash de contraseñas

Las contraseñas nunca se guardan en texto plano. Se hashean con
algoritmos lentos como bcrypt (passlib) y la verificación se hace contra
el hash:

```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

hash_guardado = pwd_context.hash("mi-clave")
pwd_context.verify("mi-clave", hash_guardado)  # True
```

## Scopes

Los scopes son permisos declarados en el token OAuth2.
`OAuth2SecurityScopes` permite validar en el endpoint qué scopes debe
tener el token para autorizar la operación: el login emite un token con
los scopes del usuario y cada endpoint declara los que exige, lo que
habilita autorización fina (lectura vs escritura) sobre la misma
autenticación.
