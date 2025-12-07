# src/monitoring/monitor.py
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import pandas as pd

def monitor_drift(reference_path='data/processed/train.csv', current_path='data/processed/test.csv', output_path='reports/'):
    ref_data = pd.read_csv(reference_path)
    curr_data = pd.read_csv(current_path)
    
    dashboard = Dashboard(tabs=[DataDriftTab()])
    dashboard.calculate(ref_data.drop('SalePrice', axis=1), curr_data.drop('SalePrice', axis=1))
    dashboard.save(f"{output_path}data_drift_report.html")
    print("Drift report generated.")

if __name__ == "__main__":
    monitor_drift()