"""Catálogo in-memory de clientes y pedidos para el agente ReAct."""

from __future__ import annotations

CLIENTES: dict[int, dict] = {
    101: {
        "id": 101,
        "nombre": "Ana López",
        "email": "ana.lopez@example.com",
        "telefono": "+54 11 5555-0101",
    },
    102: {
        "id": 102,
        "nombre": "Carlos Méndez",
        "email": "carlos.mendez@example.com",
        "telefono": "+54 11 5555-0102",
    },
}

PEDIDOS: dict[int, list[dict]] = {
    101: [
        {"id": 401, "total": 8200, "fecha": "2024-02-01", "estado": "entregado"},
        {"id": 402, "total": 3100, "fecha": "2024-05-12", "estado": "entregado"},
    ],
    102: [
        {"id": 501, "total": 3000, "fecha": "2024-01-10", "estado": "entregado"},
        {"id": 502, "total": 5500, "fecha": "2024-03-15", "estado": "entregado"},
        {"id": 503, "total": 6000, "fecha": "2024-06-20", "estado": "entregado"},
    ],
}
