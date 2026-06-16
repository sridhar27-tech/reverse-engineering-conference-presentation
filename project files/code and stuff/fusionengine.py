class FusionEngine:
    def __init__(self, alpha=0.58):
        self.alpha = alpha  # Weight for EEG from the paper
    
    def fuse_predictions(self, p_eeg: float, p_nlp: float) -> float:
        """Decision-level late fusion"""
        return self.alpha * p_eeg + (1 - self.alpha) * p_nlp
    
    def get_risk_category(self, risk_score: float) -> str:
        if risk_score < 0.3:
            return "No Risk"
        elif risk_score < 0.6:
            return "Low Risk"
        elif risk_score < 0.85:
            return "Moderate Risk"
        else:
            return "High Risk / Active Intent"