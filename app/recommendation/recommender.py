"""Recommendation engine: semantic neighbors + category affinity + cluster membership + citizen journeys."""

from __future__ import annotations

import structlog

from app.config import CONFIG
from app.indexing.cluster_builder import ClusterIndex
from app.indexing.vector_index import VectorIndex
from app.models import RecommendedService, Service

logger = structlog.get_logger()

# ── Citizen journey map ───────────────────────────────────────────────────
# Hand-curated service connections that represent real citizen journeys.
# These capture relationships that semantic similarity alone cannot:
# e.g., "pregnant woman" needs maternity + baby kit + Bolsa Família,
# but these services are not lexically/semantically close.
CITIZEN_JOURNEYS: dict[str, list[tuple[str, str]]] = {
    # Pregnancy journey
    "atendimento-em-maternidades-cffe0736": [
        ("distribuicao-de-kit-enxoval-do-bebe-77f09458", "jornada gestante: kit enxoval"),
        ("informacoes-sobre-o-programa-bolsa-familia-4547c2ba", "jornada gestante: apoio financeiro"),
        ("informacoes-sobre-vacinacao-humana-728a6848", "jornada gestante: vacinação do bebê"),
    ],
    "distribuicao-de-kit-enxoval-do-bebe-77f09458": [
        ("atendimento-em-maternidades-cffe0736", "jornada gestante: maternidade"),
        ("informacoes-sobre-o-programa-bolsa-familia-4547c2ba", "jornada gestante: apoio financeiro"),
    ],
    # Tax/property journey
    "emissao-de-2-via-do-iptu-ce2b748c": [
        ("iptu-consulta-a-pagamentos-e-debito-automatico-b175364b", "jornada tributária: consultar débitos"),
        ("parcelamento-de-debitos-em-divida-ativa-6ba1f0f4", "jornada tributária: parcelar dívida"),
        ("certidao-negativa-de-debito-nada-consta-439306e1", "jornada tributária: certidão negativa"),
    ],
    "iptu-consulta-a-pagamentos-e-debito-automatico-b175364b": [
        ("emissao-de-2-via-do-iptu-ce2b748c", "jornada tributária: emitir boleto"),
        ("parcelamento-de-debitos-em-divida-ativa-6ba1f0f4", "jornada tributária: parcelar dívida"),
    ],
    "parcelamento-de-debitos-em-divida-ativa-6ba1f0f4": [
        ("certidao-negativa-de-debito-nada-consta-439306e1", "jornada tributária: certidão negativa"),
        ("iptu-consulta-a-pagamentos-e-debito-automatico-b175364b", "jornada tributária: consultar pagamentos"),
    ],
    # Property regularization journey
    "certidao-de-habite-se-aceitacao-df83d300": [
        ("informacoes-sobre-cadastro-no-programa-minha-casa-401628a4", "jornada imóvel: programa habitacional"),
        ("certidao-negativa-de-debito-nada-consta-439306e1", "jornada imóvel: certidão negativa"),
    ],
    # Animal care journey
    "atendimento-clinico-em-animais-8c9a32e8": [
        ("castracao-gratuita-de-caes-e-gatos-programa-bicho-797d5e5f", "jornada animal: castração"),
        ("cadastro-de-animais-no-sisbicho-b5ad2d27", "jornada animal: cadastro SISBICHO"),
    ],
    "castracao-gratuita-de-caes-e-gatos-programa-bicho-797d5e5f": [
        ("cadastro-de-animais-no-sisbicho-b5ad2d27", "jornada animal: cadastro SISBICHO"),
        ("atendimento-clinico-em-animais-8c9a32e8", "jornada animal: atendimento clínico"),
    ],
    # Education journey
    "informacoes-sobre-matricula-na-rede-municipal-2026-6c635361": [
        ("informacoes-sobre-merenda-escolar-146237e8", "jornada escolar: merenda"),
        ("inclusao-de-aluno-para-acompanhamento-escolar-b1ed4c9e", "jornada escolar: acompanhamento"),
    ],
    # Health journey
    "atendimento-em-unidades-de-atencao-primaria-em-2f6e4910": [
        ("atendimento-em-unidades-de-pronto-atendimento-upa-362ec1a2", "jornada saúde: UPA"),
        ("informacoes-sobre-vacinacao-humana-728a6848", "jornada saúde: vacinação"),
        ("distribuicao-de-insumos-para-tratamento-de-7e3ea1a4", "jornada saúde: insumos"),
    ],
    # Employment journey
    "consulta-e-encaminhamento-para-vagas-de-emprego-a8a12ae6": [
        ("inclusao-de-pessoas-com-deficiencia-no-mercado-de-2b6c31d8", "jornada emprego: PcD"),
        ("informacoes-sobre-educacao-de-jovens-e-adultos-eja-901bf85b", "jornada emprego: qualificação EJA"),
    ],
    # Vulnerability journey
    "atendimento-para-pessoas-vitimas-de-violencia-2edbfc24": [
        ("informacoes-sobre-acoes-de-acolhimento-a-pessoas-8aaba05a", "jornada acolhimento: população de rua"),
    ],
    "informacoes-sobre-acoes-de-acolhimento-a-pessoas-8aaba05a": [
        ("cadastro-para-acesso-as-cozinhas-comunitarias-042e8b69", "jornada acolhimento: alimentação"),
        ("consulta-e-encaminhamento-para-vagas-de-emprego-a8a12ae6", "jornada acolhimento: emprego"),
    ],
}

JOURNEY_BOOST = 0.15


class Recommender:
    """Generate service recommendations based on search results."""

    def __init__(
        self,
        services_map: dict[str, Service],
        vector_index: VectorIndex,
        cluster_index: ClusterIndex,
    ) -> None:
        self._services = services_map
        self._vector_index = vector_index
        self._cluster_index = cluster_index

    def recommend(
        self,
        result_ids: list[str],
        top_k: int | None = None,
    ) -> list[RecommendedService]:
        """Generate recommendations based on the top search results.

        Scoring (4 signals):
          - semantic_similarity to search results (primary signal)
          - same tema_geral bonus
          - same semantic cluster bonus
          - citizen journey bonus (hand-curated service connections)
        """
        top_k = top_k or CONFIG.rec_max_results
        if not result_ids:
            return []

        exclude = set(result_ids)
        candidate_scores: dict[str, float] = {}
        candidate_reasons: dict[str, list[str]] = {}

        seed_ids = result_ids[:CONFIG.rec_seed_count]

        for seed_id in seed_ids:
            # Strategy A: semantic neighbors
            neighbors = self._vector_index.get_neighbors(
                seed_id, top_k=CONFIG.rec_semantic_neighbors
            )
            for neighbor_id, sim_score in neighbors:
                if neighbor_id in exclude:
                    continue

                score = sim_score
                reasons = [f"similar a '{self._services[seed_id].nome}'"]

                # Category boost
                neighbor_svc = self._services.get(neighbor_id)
                seed_svc = self._services.get(seed_id)
                same_category = neighbor_svc and seed_svc and neighbor_svc.tema == seed_svc.tema

                # Filter out low-similarity cross-category noise
                if not same_category and sim_score < CONFIG.rec_cross_category_min_sim:
                    continue

                if same_category:
                    score += CONFIG.rec_category_boost
                    reasons.append(f"mesma categoria ({neighbor_svc.tema})")

                # Cluster boost
                if self._cluster_index.same_cluster(seed_id, neighbor_id):
                    score += CONFIG.rec_cluster_boost
                    reasons.append("mesmo grupo temático")

                # Keep best score across seeds
                if neighbor_id not in candidate_scores or score > candidate_scores[neighbor_id]:
                    candidate_scores[neighbor_id] = score
                    candidate_reasons[neighbor_id] = reasons

            # Strategy B: citizen journey connections
            journey_links = CITIZEN_JOURNEYS.get(seed_id, [])
            for linked_id, journey_reason in journey_links:
                if linked_id in exclude or linked_id not in self._services:
                    continue
                journey_score = candidate_scores.get(linked_id, CONFIG.rec_cross_category_min_sim) + JOURNEY_BOOST
                journey_reasons = candidate_reasons.get(linked_id, [])
                if journey_reason not in journey_reasons:
                    journey_reasons = journey_reasons + [journey_reason]
                if linked_id not in candidate_scores or journey_score > candidate_scores[linked_id]:
                    candidate_scores[linked_id] = journey_score
                    candidate_reasons[linked_id] = journey_reasons

        # Sort by score and take top-k
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

        recommendations = []
        for cand_id, score in sorted_candidates[:top_k]:
            service = self._services.get(cand_id)
            if service:
                reason_parts = candidate_reasons.get(cand_id, [])
                reason_text = "; ".join(reason_parts) if reason_parts else "serviço relacionado"
                recommendations.append(
                    RecommendedService(service=service, score=round(score, 4), reason=reason_text)
                )

        logger.debug(
            "recommendations_generated",
            seed_count=len(seed_ids),
            candidates_evaluated=len(candidate_scores),
            returned=len(recommendations),
        )

        return recommendations
