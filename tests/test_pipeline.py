"""Tests unitaires basiques — ne nécessitent pas OpenSearch ni LLM."""
import sys, os
sys.path.insert(0, os.path.abspath("."))

from src.llm.rag_pipeline import to_rag_tickets
from src.llm.llm_client import build_context, build_prompt


def test_to_rag_tickets_empty():
    assert to_rag_tickets([]) == []


def test_to_rag_tickets_parsing():
    fake = [{"_source": {"text": "Sujet | Corps du ticket | Résolution proposée",
                         "language": "fr", "type": "incident", "priority": "high"}}]
    result = to_rag_tickets(fake)
    assert len(result) == 1
    assert result[0]["subject"] == "Sujet"
    assert result[0]["body_preview"] == "Corps du ticket"
    assert result[0]["answer"] == "Résolution proposée"
    assert result[0]["language"] == "fr"


def test_build_context_empty():
    assert build_context([]) == ""


def test_build_prompt_structure():
    tickets = [{"subject": "Test", "body_preview": "Corps", "answer": "Réponse"}]
    messages = build_prompt("Ma question", tickets, lang="fr")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Ma question" in messages[1]["content"]