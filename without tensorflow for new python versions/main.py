import numpy as np
import joblib
from nlp_module.simple_nlp import predict_suicide_risk
from fusion.fusion_engine import FusionEngine
from utils.report_generator import generate_report   # ← Added

def load_eeg_model():
    try:
        model = joblib.load('models/eeg_model.pkl')
        scaler = joblib.load('models/eeg_scaler.pkl')
        return model, scaler
    except:
        return None, None

def main():
    print("=== Unified EEG + NLP Mental Health Framework ===\n")
    
    # EEG
    model, scaler = load_eeg_model()
    sample_eeg = np.random.rand(1, 500)
    p_eeg = model.predict_proba(scaler.transform(sample_eeg))[0][1] if model else 0.58
    
    print(f"EEG Depression Probability : {p_eeg:.4f}")
    
    # NLP
    sample_text = "I feel completely hopeless and have been thinking about ending everything lately."
    p_nlp = predict_suicide_risk(sample_text)
    print(f"NLP Suicide Risk Probability: {p_nlp:.4f}")
    
    # Fusion
    fusion = FusionEngine(alpha=0.58)
    risk_score = fusion.fuse_predictions(p_eeg, p_nlp)
    category = fusion.get_risk_category(risk_score)
    
    print(f"\nFused Risk Score          : {risk_score:.4f}")
    print(f"Clinical Risk Category    : {category}")
    
    # Generate Report
   # ... existing code ...

    # Generate Reports (PDF + HTML)
    generate_report(p_eeg, p_nlp, risk_score, category, sample_text)
    
    print("\n✅ Done! Check 'risk_report.txt' in the project folder.")

if __name__ == "__main__":
    main()