"""Query preprocessing: normalization, local synonym expansion, optional LLM enrichment."""

from __future__ import annotations

import os
import re

import structlog
from unidecode import unidecode

logger = structlog.get_logger()


# ── Local query expansion ─────────────────────────────────────────────────
# Lightweight, zero-latency synonym expansion for known vocabulary gaps.
# Maps normalized patterns to additional terms appended to the query.
# These address failure cases identified in the evaluation cycle:
#   - "árvore caiu" → needs "remoção" (causal inference gap)
#   - "refeição gratuita / fome" → needs "cozinha comunitária" (semantic confusion with "gratuito")
#   - "comer de graça" → same issue
# Each entry is (pattern, expansion, anti_patterns_or_None).
# When anti_patterns is set and ANY anti-pattern appears in the query,
# the expansion is skipped — prevents collisions like "caindo" matching
# both "caí da escada" and "barranco caindo".
SynonymEntry = tuple[str, str, frozenset[str] | None]

SYNONYM_EXPANSIONS: list[SynonymEntry] = [
    # ── Health emergencies ──────────────────────────────────────────────
    ("quebrei", "hospital emergencia upa fratura atendimento medico", None),
    ("fratura", "hospital emergencia upa atendimento medico", None),
    ("acidente", "hospital emergencia upa pronto atendimento", None),
    ("machucado", "hospital emergencia upa atendimento", None),
    ("machucou", "hospital emergencia upa atendimento", None),
    ("machuquei", "hospital emergencia upa atendimento", None),
    ("sangrando", "hospital emergencia upa atendimento", None),
    ("ambulancia", "hospital emergencia upa pronto atendimento", None),
    ("passando mal", "hospital emergencia upa pronto atendimento", None),
    ("desmaio", "hospital emergencia upa pronto atendimento", None),
    ("desmaiou", "hospital emergencia upa pronto atendimento", None),
    ("infarto", "hospital emergencia upa pronto atendimento", None),
    ("convulsao", "hospital emergencia upa pronto atendimento", None),
    ("dor no peito", "hospital emergencia upa pronto atendimento cardiaco", None),
    ("dor forte", "hospital emergencia upa pronto atendimento", None),
    # "caindo" only for health context — NOT barranco/morro/terra/arvore
    ("caindo", "hospital emergencia upa pronto atendimento",
     frozenset({"barranco", "morro", "terra", "encosta", "barreira", "arvore", "galho"})),
    ("cai da escada", "hospital emergencia upa pronto atendimento", None),
    ("cai e bateu", "hospital emergencia upa pronto atendimento", None),
    ("bateu a cabeca", "hospital emergencia upa pronto atendimento", None),
    ("atropelado", "hospital emergencia upa pronto atendimento", frozenset({"cachorro", "gato", "animal"})),
    # ── Illness / fever ─────────────────────────────────────────────────
    ("febre", "upa pronto atendimento saude unidade atencao primaria", None),
    ("gripado", "upa pronto atendimento saude unidade", None),
    ("gripe", "upa pronto atendimento saude unidade", None),
    ("resfriado", "upa pronto atendimento saude unidade", None),
    ("tosse", "upa pronto atendimento saude unidade", None),
    ("vomito", "upa pronto atendimento saude unidade", None),
    ("vomitou", "upa pronto atendimento saude unidade", None),
    ("vomitando", "upa pronto atendimento saude unidade", None),
    ("diarreia", "upa pronto atendimento saude unidade", None),
    ("dor de cabeca", "upa pronto atendimento saude unidade", None),
    ("dor de barriga", "upa pronto atendimento saude unidade", None),
    ("doente", "upa pronto atendimento saude unidade atencao primaria", frozenset({"cachorro", "gato", "animal"})),
    # ── Health / primary care ──────────────────────────────────────────
    ("consulta no postinho", "atencao primaria saude unidade clinica familia", None),
    ("consulta no posto", "atencao primaria saude unidade clinica familia", None),
    ("consulta no sus", "atencao primaria saude unidade clinica familia", None),
    ("marcar consulta", "atencao primaria saude unidade clinica familia", frozenset({"multa", "transito", "emprego"})),
    ("postinho", "atencao primaria saude unidade clinica familia", None),
    # ── Medicine / treatment ────────────────────────────────────────────
    ("remedio", "insumos saude farmacia atendimento medicamento", None),
    ("medicamento", "insumos saude farmacia atendimento", None),
    ("insulina", "diabetes insumos tratamento distribuicao", None),
    ("parar de fumar", "antitabagismo tratamento programa", None),
    # ── Dental ──────────────────────────────────────────────────────────
    ("dentista", "atendimento saude bucal odontologico clinica", None),
    ("dente", "atendimento saude bucal odontologico clinica", None),
    ("dor de dente", "atendimento saude bucal odontologico clinica", None),
    # ── Security / violence ─────────────────────────────────────────────
    ("assaltado", "vitima violencia seguranca atendimento", None),
    ("assalto", "vitima violencia seguranca atendimento", None),
    ("roubo", "vitima violencia seguranca atendimento", None),
    ("furto", "vitima violencia seguranca atendimento", None),
    ("violencia domestica", "vitima violencia atendimento mulher", None),
    ("agredido", "vitima violencia atendimento", None),
    ("agredida", "vitima violencia atendimento", None),
    ("me bateu", "vitima violencia atendimento", None),
    ("apanhando", "vitima violencia domestica atendimento", frozenset({"cachorro", "gato", "animal"})),
    ("sofrendo agressao", "vitima violencia domestica atendimento", None),
    ("agressao", "vitima violencia atendimento", frozenset({"cachorro", "gato", "animal"})),
    ("bate na esposa", "vitima violencia domestica atendimento mulher", None),
    ("bate na mulher", "vitima violencia domestica atendimento mulher", None),
    # ── Animal welfare ──────────────────────────────────────────────────
    ("bate no cachorro", "maus tratos animais vistoria denuncia", None),
    ("bate no gato", "maus tratos animais vistoria denuncia", None),
    ("maltrata", "maus tratos animais vistoria denuncia", None),
    ("maus tratos", "vistoria animais denuncia", None),
    ("judia do animal", "maus tratos animais vistoria denuncia", None),
    ("judia do cachorro", "maus tratos animais vistoria denuncia", None),
    ("cachorro preso", "maus tratos animais vistoria denuncia", None),
    ("animal preso", "maus tratos animais vistoria denuncia", None),
    ("cachorro atropelado", "clinico animais veterinario atendimento", None),
    ("gato atropelado", "clinico animais veterinario atendimento", None),
    ("documento pro meu cachorro", "sisbicho cadastro animais registro", None),
    ("documento pro meu gato", "sisbicho cadastro animais registro", None),
    ("registrar meu cachorro", "sisbicho cadastro animais registro", None),
    ("registrar meu gato", "sisbicho cadastro animais registro", None),
    # ── Trees / vegetation ──────────────────────────────────────────────
    ("arvore caiu", "remocao de arvore vias publicas", None),
    ("arvore caida", "remocao de arvore vias publicas", None),
    ("arvore caindo", "remocao de arvore vias publicas", None),
    ("galho de arvore", "poda remocao arvore", None),
    ("galho caiu", "remocao poda arvore vias publicas", None),
    ("arvore bloqueando", "remocao de arvore vias publicas", None),
    ("arvore no fio", "remocao poda arvore vias publicas", None),
    # ── Food access ─────────────────────────────────────────────────────
    ("fome", "cozinha comunitaria alimentacao refeicao", None),
    ("sem dinheiro", "cozinha comunitaria alimentacao bolsa familia assistencia", None),
    ("refeicao gratuita", "cozinha comunitaria", None),
    ("refeicao de graca", "cozinha comunitaria alimentacao", None),
    ("comer de graca", "cozinha comunitaria alimentacao", None),
    ("passar fome", "cozinha comunitaria alimentacao", None),
    ("refeicao barata", "cozinha comunitaria alimentacao", None),
    ("restaurante popular", "cozinha comunitaria alimentacao", None),
    # ── Wildlife ────────────────────────────────────────────────────────
    ("cobra", "resgate animal silvestre", None),
    ("serpente", "resgate animal silvestre", None),
    ("lagarto", "resgate animal silvestre", None),
    ("bicho silvestre", "resgate animal silvestre", None),
    ("jacare", "resgate animal silvestre", None),
    ("papagaio", "resgate animal silvestre ave", None),
    ("passaro", "resgate animal silvestre ave", None),
    ("coruja", "resgate animal silvestre ave", None),
    ("tucano", "resgate animal silvestre ave", None),
    ("gambazinho", "resgate animal silvestre", None),
    ("gamba", "resgate animal silvestre", frozenset({"samba"})),
    ("macaco", "resgate animal silvestre", None),
    # ── Flooding / weather / landslides ─────────────────────────────────
    ("alagando", "defesa civil alerta alagamento enchente", None),
    ("alagamento", "defesa civil alerta deslizamento enchente", None),
    ("enchente", "defesa civil alerta alagamento", None),
    ("inundacao", "defesa civil alerta alagamento", None),
    ("deslizamento", "defesa civil barreira vistoria", None),
    ("dengue no bairro", "foco aedes aegypti vistoria", None),
    ("morro rachando", "vistoria deslizamento barreira encosta", None),
    ("barranco", "deslizamento barreira vistoria encosta defesa civil", None),
    ("terra descendo", "deslizamento barreira vistoria encosta defesa civil", None),
    ("terra do morro", "deslizamento barreira vistoria encosta defesa civil", None),
    ("encosta cedendo", "deslizamento barreira vistoria encosta defesa civil", None),
    ("muro de contencao", "deslizamento barreira vistoria encosta", None),
    # ── Housing ─────────────────────────────────────────────────────────
    ("moradia popular", "minha casa minha vida programa habitacional", None),
    ("casa popular", "minha casa minha vida programa habitacional", None),
    ("mcmv", "minha casa minha vida programa habitacional cadastro", None),
    ("direito ao mcmv", "minha casa minha vida programa habitacional renda", None),
    # ── Employment ──────────────────────────────────────────────────────
    ("desempregado", "emprego vaga trabalho encaminhamento consulta", None),
    ("desempregada", "emprego vaga trabalho encaminhamento consulta", None),
    ("procurando emprego", "emprego vaga trabalho encaminhamento", None),
    ("carteira de trabalho", "emprego vaga trabalho", None),
    ("abrir empresa", "empreendedor alvara licenca", None),
    ("sem emprego", "emprego vaga trabalho encaminhamento consulta", None),
    ("sine", "emprego vaga trabalho encaminhamento", None),
    # ── Education ───────────────────────────────────────────────────────
    ("matricular meu filho", "matricula escola municipal rede ensino", None),
    ("vaga na escola", "matricula escola municipal rede ensino", None),
    ("vaga em escola", "matricula escola municipal rede ensino", None),
    ("escola publica", "matricula escola municipal rede ensino", frozenset({"merenda"})),
    ("creche", "matricula escola municipal educacao infantil", None),
    ("voltar a estudar", "educacao jovens adultos eja ensino", None),
    ("terminar os estudos", "educacao jovens adultos eja ensino", None),
    ("supletivo", "educacao jovens adultos eja ensino", None),
    ("eja", "educacao jovens adultos ensino", None),
    # ── Social assistance ───────────────────────────────────────────────
    ("morador de rua", "acolhimento situacao rua pessoa", None),
    ("situacao de rua", "acolhimento pessoa rua", None),
    ("morando debaixo", "acolhimento situacao rua pessoa", None),
    ("morando na rua", "acolhimento situacao rua pessoa", None),
    ("sem teto", "acolhimento situacao rua pessoa", None),
    ("abrigo", "acolhimento situacao rua pessoa", frozenset({"animal", "cachorro", "gato"})),
    ("dormindo na rua", "acolhimento situacao rua pessoa", None),
    ("beneficio social", "bolsa familia bpc assistencia", None),
    ("ajuda financeira", "bolsa familia bpc assistencia beneficio", None),
    ("cadastro unico", "bolsa familia cadunico atualizar", None),
    ("cadunico", "bolsa familia cadastro unico atualizar", None),
    # ── Elderly / disability ────────────────────────────────────────────
    ("sou idosa", "beneficio prestacao continuada bpc idoso assistencia", None),
    ("sou idoso", "beneficio prestacao continuada bpc idoso assistencia", None),
    ("nao consigo trabalhar", "beneficio prestacao continuada bpc assistencia", None),
    ("cadeira de rodas", "deficiencia pcd estacionamento cartao", None),
    ("deficiente", "pessoa deficiencia pcd inclusao", None),
    ("dinheiro do governo", "beneficio prestacao continuada bpc bolsa familia assistencia", None),
    # ── Culture / entertainment ─────────────────────────────────────────
    ("forro", "bailes populares danca evento gratuito cinelandia", frozenset({"carro", "pneu"})),
    ("forro de graca", "bailes populares danca evento gratuito", None),
    ("festa de graca", "bailes populares evento gratuito cinelandia", None),
    ("jogo de graca", "cadeira cativa ingressos gratuitos", None),
    ("ingresso gratis", "cadeira cativa ingressos gratuitos", None),
    ("levar meu filho no jogo", "cadeira cativa ingressos gratuitos", None),
    ("assistir jogo de graca", "cadeira cativa ingressos gratuitos", None),
]


def expand_query(query: str) -> str:
    """Expand query with local synonyms for known vocabulary gaps.

    Returns the original query with additional terms appended if any
    pattern matches. Supports anti-patterns: if a guard set is provided
    and any anti-pattern appears in the query, the expansion is skipped.
    """
    normalized = unidecode(query.lower())
    additions = []
    for pattern, expansion, anti_patterns in SYNONYM_EXPANSIONS:
        if pattern in normalized:
            if anti_patterns and any(ap in normalized for ap in anti_patterns):
                continue
            additions.append(expansion)

    if additions:
        expanded = f"{query} {' '.join(additions)}"
        logger.debug("query_expanded_local", original=query, expanded=expanded)
        return expanded
    return query


def normalize_query(query: str) -> str:
    """Basic normalization: strip, collapse whitespace."""
    return re.sub(r"\s+", " ", query.strip())


async def enrich_query_with_llm(query: str) -> dict:
    """Use LLM to extract intent and suggest query expansion.

    Returns dict with keys: original, expanded, intent.
    Falls back gracefully if LLM is unavailable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"original": query, "expanded": query, "intent": ""}

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        from app.config import CONFIG

        response = await client.chat.completions.create(
            model=CONFIG.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um assistente que ajuda cidadãos a encontrar serviços públicos "
                        "da Prefeitura do Rio de Janeiro. Dada uma consulta do usuário, extraia:\n"
                        "1. A intenção principal (1 frase curta)\n"
                        "2. Termos de busca expandidos em português (sinônimos, termos relacionados)\n"
                        "Responda APENAS em JSON: {\"intent\": \"...\", \"expanded\": \"...\"}"
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=CONFIG.llm_max_tokens,
            timeout=CONFIG.llm_timeout,
        )

        import json

        content = response.choices[0].message.content or ""
        # Extract JSON from response
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return {
                "original": query,
                "expanded": parsed.get("expanded", query),
                "intent": parsed.get("intent", ""),
            }
    except Exception as e:
        logger.warning("llm_enrichment_failed", error=str(e))

    return {"original": query, "expanded": query, "intent": ""}
