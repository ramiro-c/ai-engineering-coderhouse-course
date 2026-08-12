"""Paquete de clientes externos de la pre-entrega 4.

Expone la factory multi-proveedor de modelos de chat (evolución B, RF-6),
patrón de pre-entrega-3/clients: los imports de las librerías de cada
provider son lazy (dentro de build_chat_model) para que importar este
paquete no requiera credenciales ni dependencias opcionales instaladas.
"""

from clients.factory import build_chat_model

__all__ = ["build_chat_model"]
