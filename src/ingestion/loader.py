"""Module de chargement et exploration des données brutes."""

import pandas as pd
from loguru import logger
from pathlib import Path


def load_tickets(filepath: str | Path) -> pd.DataFrame:
    """Charge le dataset de tickets SAV depuis un fichier CSV."""
    filepath = Path(filepath)
    logger.info(f"Chargement de {filepath}...")

    df = pd.read_csv(filepath, low_memory=False)
    logger.info(f"Dataset chargé : {len(df)} tickets, {len(df.columns)} colonnes")
    logger.info(f"Colonnes : {list(df.columns)}")
    return df


def explore_dataset(df: pd.DataFrame) -> dict:
    """Retourne des statistiques de base sur le dataset."""
    stats = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "sample_types": (
            df["type"].value_counts().head(10).to_dict()
            if "type" in df.columns else {}
        ),
        "sample_queues": (
            df["queue"].value_counts().head(10).to_dict()
            if "queue" in df.columns else {}
        ),
        "sample_priorities": (
            df["priority"].value_counts().to_dict()
            if "priority" in df.columns else {}
        ),
        "languages": (
            df["language"].value_counts().to_dict()
            if "language" in df.columns else {}
        ),
    }
    return stats