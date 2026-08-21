"""Paquete de clientes externos de la pre-entrega 5.

Expone la factory multi-proveedor de modelos de chat para el agente ReAct:
los imports de las librerías de cada provider son lazy (dentro de
build_chat_model) para que importar este paquete no requiera credenciales ni
dependencias opcionales instaladas.
"""

from clients.factory import build_chat_model

__all__ = ["build_chat_model"]
