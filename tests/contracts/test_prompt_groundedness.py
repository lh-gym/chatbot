from rag_agent.agent.planner import _SYSTEM_PROMPT
from rag_agent.obs.tracing import GroundednessEvaluator


def test_prompt_contains_groundedness_constraints() -> None:
    assert "Groundedness" in _SYSTEM_PROMPT
    assert "Cite every factual statement" in _SYSTEM_PROMPT


def test_groundedness_evaluator_high_for_cited_supported_answer() -> None:
    evaluator = GroundednessEvaluator(min_overlap=0.3)
    answer = "All employees must encrypt customer data at rest [policy-doc-chunk-0000]."
    sources = ["Company policy states all employees must encrypt customer data at rest."]

    assert evaluator.score(answer, sources) >= 0.95
