import pandas as pd
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
from evidently.metric_preset import DataDriftPreset
import os

def monitor_drift(
    reference_path: str = "data/processed/train.csv",
    current_path: str = "data/processed/test.csv",
    output_path: str = "reports/"
):
    
    # Créer le dossier reports s'il n'existe pas
    os.makedirs(output_path, exist_ok=True)

    # Charger les données
    ref_data = pd.read_csv(reference_path)
    curr_data = pd.read_csv(current_path)

    print(f"Reference dataset: {len(ref_data)} lignes")
    print(f"Current dataset:   {len(curr_data)} lignes")

    # Enlever la cible (SalePrice) pour comparer seulement les features
    ref_features = ref_data.drop(columns=["SalePrice"], errors="ignore")
    curr_features = curr_data.drop(columns=["SalePrice"], errors="ignore")

    # Créer le rapport avec le preset DataDrift (le plus beau et complet)
    data_drift_report = Report(metrics=[DataDriftPreset()])

    # Calculer le drift
    data_drift_report.run(reference_data=ref_features, current_data=curr_features)

    # Sauvegarder le rapport HTML
    report_path = os.path.join(output_path, "data_drift_report.html")
    data_drift_report.save_html(report_path)

    print(f"Rapport de drift généré avec succès !")
    print(f"Ouvrez ce fichier → {report_path}")

if __name__ == "__main__":
    monitor_drift()