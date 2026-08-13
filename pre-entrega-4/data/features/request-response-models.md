# Request y response models con Pydantic

Los modelos de Pydantic son la forma declarativa de definir el cuerpo de
una petición (request body) y la forma de la respuesta en FastAPI. El
framework valida, serializa y documenta automáticamente los datos, de
modo que el contrato de la API queda expresado en tipos.

## Definir un modelo

Un modelo es una clase que hereda de `BaseModel` con campos tipados:

```python
from pydantic import BaseModel

class Item(BaseModel):
    nombre: str
    precio: float
    disponible: bool = True
```

## Request body

El cuerpo de la petición se recibe declarando un parámetro del modelo en
la función. FastAPI valida que el JSON enviado cumpla los tipos:

```python
@app.post("/items")
def crear_item(item: Item):
    return {"nombre": item.nombre, "precio": item.precio}
```

Si falta un campo obligatorio o el tipo no coincide, la respuesta es un
error de validación 422 con el detalle de cada campo: el nombre del
campo, el motivo del rechazo y la ubicación en el cuerpo.

## Validación de campos

Pydantic valida tipos y además se pueden agregar restricciones con
`Field`: mínimo y máximo para números, longitud para strings, y
expresiones regulares:

```python
from pydantic import Field

class Item(BaseModel):
    nombre: str = Field(min_length=1, max_length=50)
    precio: float = Field(gt=0)
```

| Restricción | Aplica a | Efecto |
|---|---|---|
| `min_length` / `max_length` | strings | Límite de caracteres |
| `gt` / `ge` / `lt` / `le` | números | Mayor que, mayor o igual, menor que, menor o igual |
| `pattern` | strings | Expresión regular |
| `min_items` / `max_items` | listas | Cantidad de elementos |

## Response model

El parámetro `response_model` del decorador declara el esquema de salida.
FastAPI filtra los campos de la respuesta que no están en el modelo y
documenta el esquema en OpenAPI:

```python
@app.post("/items", response_model=Item)
def crear_item(item: Item):
    return item
```

## Modelos de entrada y salida separados

Es una buena práctica usar modelos distintos para entrada y salida: el de
entrada admite campos opcionales que la salida exige, o la salida oculta
campos internos como `password` o `token`:

```python
class ItemEntrada(BaseModel):
    nombre: str
    password: str

class ItemSalida(BaseModel):
    nombre: str
    id: int

@app.post("/items", response_model=ItemSalida)
def crear_item(item: ItemEntrada):
    return {"nombre": item.nombre, "id": 1}
```

La separación evita exponer datos sensibles y desacopla el formato de la
entrada del de la respuesta.

## Campos opcionales y defaults

Los campos con valor por defecto o tipados como `None` son opcionales en
la entrada. El default se usa cuando el cliente no envía el campo. Esto
permite evolucionar el modelo sin romper a los clientes: agregar un campo
opcional es un cambio compatible, mientras que volver obligatorio un campo
existente rompe el contrato. Los modelos anidados (un campo cuyo tipo es
otro `BaseModel`) se validan recursivamente, lo que permite expresar
jerarquías completas de datos en una sola declaración.
