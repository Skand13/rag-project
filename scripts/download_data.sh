#!/bin/bash
set -e

echo "Téléchargement du dataset Kaggle..."
source .env

kaggle datasets download -d tobiasbueck/multilingual-customer-support-tickets \
  --path data/raw/ \
  --unzip

echo "Téléchargement terminé. Fichiers disponibles dans data/raw/"
ls -lh data/raw/