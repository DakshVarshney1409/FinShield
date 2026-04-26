import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def check_data_drift(reference_data_path, current_data_path):
    """
    Uses Evidently AI to detect if production data has drifted 
    from the original training distribution.
    """
    ref_data = pd.read_csv(reference_data_path)
    cur_data = pd.read_csv(current_data_path)

    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=ref_data, current_data=cur_data)
    
    report_json = drift_report.json()
    
    # Check if 'dataset_drift' is True
    drift_detected = drift_report.as_dict()['metrics'][0]['result']['dataset_drift']
    
    if drift_detected:
        print("CRITICAL: Data drift detected. Triggering retraining...")
        # In a real scenario, this would send a signal to GitHub Actions
        return True
    return False

if __name__ == "__main__":
    check_data_drift("data/train_ref.csv", "data/production_logs.csv")
