"""Load and parse the services catalog JSON."""

from __future__ import annotations

import json
from pathlib import Path

from app.models import Service


def load_services(path: Path) -> list[Service]:
    """Parse servicos_selecionados.json into Service objects."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    services: list[Service] = []
    seen_names: set[str] = set()
    for entry in raw:
        data = json.loads(entry["data"]) if "data" in entry else entry

        nome = data.get("nome_servico", "")
        if nome in seen_names:
            continue
        seen_names.add(nome)

        service = Service(
            id=data.get("slug", ""),
            nome=nome,
            resumo=data.get("resumo_plaintext", "") or "",
            descricao_completa=data.get("descricao_completa_plaintext", "") or "",
            tema=data.get("tema_geral", "") or "",
            orgao_gestor=data.get("orgao_gestor", []) or [],
            custo=data.get("custo_servico", "") or "",
            publico=data.get("publico_especifico", []) or [],
            tempo_atendimento=data.get("tempo_atendimento", "") or "",
            instrucoes=data.get("instrucoes_solicitante_plaintext", "") or "",
            resultado=data.get("resultado_solicitacao_plaintext", "") or "",
        )
        services.append(service)

    return services
