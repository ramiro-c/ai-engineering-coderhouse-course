# Request y response models con Pydantic

Los modelos de Pydantic son la forma declarativa de definir el cuerpo de
una petición (request body) y la forma de la respuesta en FastAPI. El
framework valida, serializa y documenta automáticamente los datos.

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
error de validación 422 con el detalle de cada campo.

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

## Response model

El parámetro `response_model` del decorador declara el esquema de salida.
FastAPI filtra los campos de la respuesta que no están en el modelo y
documenta el esquema en OpenAPI:

```python
@app.post("/items", response_model=Item)
def crear_item(item: Item):
    return item
```

## Campos opcionales y defaults

Los campos con valor por defecto o tipados como `None` son opcionales en
la entrada. El default se usa cuando el cliente no envía el campo. Esto
permite evolucionar el modelo sin romper a los clientes.
