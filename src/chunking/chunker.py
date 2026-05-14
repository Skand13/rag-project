"""Module de découpage des tickets en chunks indexables."""

from dataclasses import dataclass, field
from loguru import logger
import pandas as pd


@dataclass
class Chunk:
    """Représente un chunk prêt à être indexé."""
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def build_chunk_text(row: pd.Series) -> str:
    """Construit le texte d'un chunk à partir d'une ligne du dataframe."""
    parts = []

    if pd.notna(row.get("subject")) and str(row["subject"]).strip():
        parts.append(f"Subject: {row['subject'].strip()}")

    if pd.notna(row.get("body")) and str(row["body"]).strip():
        parts.append(f"Body: {row['body'].strip()}")

    if pd.notna(row.get("answer")) and str(row["answer"]).strip():
        parts.append(f"Answer: {row['answer'].strip()}")

    return "\n\n".join(parts)


def build_metadata(row: pd.Series, idx: int) -> dict:
    """Extrait les métadonnées d'une ligne pour le filtrage OpenSearch."""
    return {
        "chunk_id":  f"ticket_{idx}",
        "type":      str(row.get("type", "")).strip(),
        "queue":     str(row.get("queue", "")).strip(),
        "priority":  str(row.get("priority", "")).strip(),
        "language":  str(row.get("language", "")).strip(),
        "version":   int(row["version"]) if pd.notna(row.get("version")) else None,
        "tags":      row.get("tags", []),
    }


def chunk_dataframe(df: pd.DataFrame) -> list[Chunk]:
    """
    Convertit chaque ligne du dataframe en un Chunk indexable.
    Stratégie : 1 ticket = 1 chunk (subject + body + answer).
    """
    chunks = []

    for idx, row in df.iterrows():
        text = build_chunk_text(row)
        if not text.strip():
            logger.warning(f"Ligne {idx} ignorée : texte vide.")
            continue

        metadata = build_metadata(row, idx)
        chunk = Chunk(
            chunk_id=metadata["chunk_id"],
            text=text,
            metadata=metadata,
        )
        chunks.append(chunk)

    logger.info(f"{len(chunks)} chunks générés depuis {len(df)} tickets.")
    return chunks