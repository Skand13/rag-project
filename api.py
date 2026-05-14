import requests
import json
import os

API_KEY = os.getenv("OPENROUTER_API_KEY", "")
url = "https://openrouter.ai/api/v1/models"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

print("Début récupération des modèles...")

try:
    response = requests.get(url, headers=headers, timeout=30)

    print("Status code :", response.status_code)

    if response.status_code != 200:
        print("Erreur API :", response.text)
        exit()

    data = response.json().get("data", [])

    generation_models = []
    embedding_models = []

    for model in data:

        model_id = model.get("id", "").lower()

        architecture = model.get("architecture", {})
        modality = architecture.get("modality", "")

        supported_parameters = model.get("supported_parameters", [])

        # détection embeddings
        if (
            "embedding" in model_id
            or "embed" in model_id
            or modality == "embedding"
        ):
            embedding_models.append(model.get("id"))

        else:
            generation_models.append(model.get("id"))

    # supprimer doublons + trier
    generation_models = sorted(list(set(generation_models)))
    embedding_models = sorted(list(set(embedding_models)))

    # sauvegarde modèles génération
    with open("generation_models.txt", "w", encoding="utf-8") as f:
        f.write("MODELES DE GENERATION\n")
        f.write("=" * 50 + "\n\n")

        for model in generation_models:
            f.write(model + "\n")

    # sauvegarde modèles embeddings
    with open("embedding_models.txt", "w", encoding="utf-8") as f:
        f.write("MODELES D'EMBEDDINGS\n")
        f.write("=" * 50 + "\n\n")

        for model in embedding_models:
            f.write(model + "\n")

    print("\n✅ Fichiers générés :")
    print("- generation_models.txt")
    print("- embedding_models.txt")

    print("\n📊 Statistiques :")
    print("Modèles génération :", len(generation_models))
    print("Modèles embeddings :", len(embedding_models))

except Exception as e:
    print("Erreur Python :", repr(e))

