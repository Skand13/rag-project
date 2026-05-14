"""Module de nettoyage et prétraitement des tickets."""

import re
import pandas as pd
from loguru import logger


# Colonnes texte du nouveau dataset
TEXT_COLUMNS = ["subject", "body", "answer"]

# Colonnes obligatoires pour qu'un ticket soit valide
REQUIRED_COLUMNS = ["body"]


def clean_text(text: str) -> str:
    """Nettoie le texte brut d'un ticket."""
    if not isinstance(text, str):
        return ""
    # Suppression des balises HTML
    text = re.sub(r'<[^>]+>', '', text)
    # Normalisation des espaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Suppression des caractères de contrôle
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text


def anonymize_pii(text: str) -> str:
    """Pseudonymise les PII résiduelles dans le texte.
    
    Le nouveau dataset ne contient pas de colonnes PII dédiées,
    mais des noms/emails peuvent apparaître dans le corps des tickets.
    Le placeholder <name> est déjà utilisé dans le dataset — on le conserve.
    """
    # Masquage des emails
    text = re.sub(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        '[EMAIL]', text
    )
    # Masquage des numéros de téléphone (formats FR et international)
    text = re.sub(
        r'\b(?:\+?33|0)[1-9](?:[.\-\s]?\d{2}){4}\b',
        '[PHONE]', text
    )
    # Masquage des numéros de carte bancaire
    text = re.sub(
        r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',
        '[CARD]', text
    )
    return text


def build_tag_list(row: pd.Series) -> list[str]:
    """Extrait la liste des tags non-nuls d'une ligne."""
    tags = []
    for i in range(1, 9):
        col = f"tag_{i}"
        if col in row.index and pd.notna(row[col]):
            tags.append(str(row[col]))
    return tags


def preprocess_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet de nettoyage du dataframe de tickets."""
    logger.info("Démarrage du prétraitement...")

    df = df.copy()

    # --- Nettoyage des colonnes texte ---
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(clean_text).apply(anonymize_pii)
            logger.info(f"Colonne '{col}' nettoyée.")

    # --- Normalisation des colonnes catégorielles ---
    for col in ["type", "queue", "priority", "language"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower()

    # --- Construction de la colonne tags (liste consolidée) ---
    df["tags"] = df.apply(build_tag_list, axis=1)

    # --- Suppression des doublons ---
    initial_len = len(df)
    df = df.drop_duplicates(subset=["subject", "body"])
    logger.info(f"Doublons supprimés : {initial_len - len(df)}")

    # --- Suppression des tickets sans corps ---
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[df["body"].str.strip() != ""]
    logger.info(f"Tickets valides après nettoyage : {len(df)}")

    return df