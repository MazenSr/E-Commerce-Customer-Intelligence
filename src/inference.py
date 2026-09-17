import joblib
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

class SegmentPredictor:

    def __init__(self, model_dir: Path = MODEL_DIR):

        self.clustering_pipe = joblib.load(model_dir / "clustering_inference_pipeline.joblib")
        self.anomaly_pipe = joblib.load(model_dir / "anomaly_inference_pipeline.joblib")

    def process_batch(self,raw_data: pd.DataFrame) -> pd.DataFrame:

        results_df = raw_data.copy()

        # Flag Anomalies
        anomaly_predictions = self.anomaly_pipe.predict(raw_data)
        results_df["Is_Anomaly"] = (anomaly_predictions == -1)

        # Calculate Anomaly Score 
        # Isolation Forest: higher decision_function = more normal
        # Inverted: higher score = more anomalous
        results_df["Anomaly_Score"] = (
            -self.anomaly_pipe.decision_function(raw_data)
        )

        # Assign Clusters
        results_df["Cluster"] = self.clustering_pipe.predict(raw_data)

        return results_df
