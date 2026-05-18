import pandas as pd
import re
import unicodedata
from pathlib import Path
from langdetect import detect


DATA_FOLDER = r"C:\Users\sofiahp\Desktop\Projet-rag\archive"
OUTPUT_FILE = r"C:\Users\sofiahp\Desktop\Projet-rag\archiveclean_tickets.csv"


def remove_accents(text):
    """Supprime les accents."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")


def clean_text(text):
    """Nettoie le texte du ticket."""

    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = remove_accents(text)

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"ticket\s?#?\d+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def detect_language(text):
    """Détecte la langue du ticket."""

    try:
        if len(text.strip()) < 5:
            return "unknown"
        return detect(text)
    except:
        return "unknown"


def detect_intent(text):
    """Détecte l'objectif principal du ticket."""

    text = text.lower()

    intents = {
        "login_issue": ["login", "password", "connexion", "mot de passe", "authentication"],
        "network_issue": ["wifi", "internet", "vpn", "network", "reseau"],
        "hardware_issue": ["printer", "screen", "keyboard", "ordinateur", "ecran"],
        "software_issue": ["bug", "crash", "application", "logiciel", "software"],
        "email_issue": ["email", "outlook", "smtp", "mail"],
    }

    for intent, keywords in intents.items():
        for keyword in keywords:
            if keyword in text:
                return intent

    return "other"


def detect_priority(text):
    """Détecte la priorité du ticket."""

    text = text.lower()

    high_words = ["urgent", "critical", "asap", "bloquant", "critique"]
    medium_words = ["problem", "issue", "lent", "erreur", "slow"]

    if any(word in text for word in high_words):
        return "high"

    if any(word in text for word in medium_words):
        return "medium"

    return "low"


def find_column(df, possible_columns):
    """Trouve une colonne existante dans le CSV."""

    for col in possible_columns:
        if col in df.columns:
            return col

    return None


def normalize_dataframe(df, source_file):
    """Normalise un fichier CSV de tickets."""

    df.columns = [col.lower().strip() for col in df.columns]

    title_col = find_column(df, ["title", "subject", "ticket_title", "titre"])
    desc_col = find_column(df, ["description", "body", "message", "content", "contenu"])

    if title_col is None:
        df["title"] = ""
        title_col = "title"

    if desc_col is None:
        df["description"] = ""
        desc_col = "description"

    df["raw_text"] = (
        df[title_col].fillna("").astype(str)
        + " "
        + df[desc_col].fillna("").astype(str)
    )

    df["clean_text"] = df["raw_text"].apply(clean_text)
    df["language"] = df["clean_text"].apply(detect_language)
    df["intent"] = df["clean_text"].apply(detect_intent)
    df["priority"] = df["clean_text"].apply(detect_priority)
    df["text_length"] = df["clean_text"].apply(len)
    df["source_file"] = source_file

    df = df[df["clean_text"].str.len() > 10]
    df = df.drop_duplicates(subset=["clean_text"])

    return df[
        [
            "raw_text",
            "clean_text",
            "language",
            "intent",
            "priority",
            "text_length",
            "source_file",
        ]
    ]


def load_all_csv():
    """Charge et normalise tous les CSV."""

    all_dataframes = []
    csv_files = list(Path(DATA_FOLDER).glob("*.csv"))

    print(f"{len(csv_files)} CSV détectés")

    for csv_file in csv_files:
        try:
            print(f"Traitement : {csv_file.name}")

            df = pd.read_csv(csv_file, encoding="utf-8", on_bad_lines="skip")
            normalized_df = normalize_dataframe(df, csv_file.name)

            all_dataframes.append(normalized_df)

            print(f"OK : {len(normalized_df)} tickets")

        except Exception as e:
            print(f"Erreur avec {csv_file.name} : {e}")

    return all_dataframes


def main():
    all_dataframes = load_all_csv()

    if not all_dataframes:
        print("Aucun CSV valide trouvé.")
        return

    final_df = pd.concat(all_dataframes, ignore_index=True)

    final_df = final_df.drop_duplicates(subset=["clean_text"])
    final_df = final_df.reset_index(drop=True)

    final_df["ticket_id"] = final_df.index + 1

    final_df = final_df[
        [
            "ticket_id",
            "raw_text",
            "clean_text",
            "language",
            "intent",
            "priority",
            "text_length",
            "source_file",
        ]
    ]

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("Normalisation terminée")
    print(f"Total tickets : {len(final_df)}")
    print(f"Fichier créé : {OUTPUT_FILE}")

    print("Langues détectées :")
    print(final_df["language"].value_counts())

    print("Intentions détectées :")
    print(final_df["intent"].value_counts())

    print("Priorités détectées :")
    print(final_df["priority"].value_counts())


if __name__ == "__main__":
    main()