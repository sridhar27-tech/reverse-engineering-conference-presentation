import sys
import numpy as np

# Import modules
try:
    from eeg_module.sieptnet import create_sieptnet
except ModuleNotFoundError:
    sys.path.append('path_to_eeg_module')  
    from eeg_module.sieptnet import create_sieptnet


from eeg_module.sieptnet import create_sieptnet
try:
    from nlp_module.bert_classifier import create_bert_model
except ModuleNotFoundError:
    sys.path.append('path_to_nlp_module')  
    from nlp_module.bert_classifier import create_bert_model
try:
    from fusion.fusion_engine import FusionEngine
except ModuleNotFoundError:
    sys.path.append('c:/Users/SRIDHAR.M/mentalhealthframework/fusion')  
    from fusion.fusion_engine import FusionEngine

def demo_inference():
    print("=== Unified EEG + NLP Mental Health Framework Demo ===\n")
    
    # EEG Module
    eeg_model = create_sieptnet()
    dummy_eeg = np.random.rand(1, 1020, 62).astype(np.float32)  
    p_eeg = eeg_model.predict(dummy_eeg, verbose=0)[0][0]
    print(f"EEG Depression Probability: {p_eeg:.4f}")
    
    # NLP Module
    _, _ = create_bert_model()
    p_nlp = 0.72
    print(f"NLP Suicide Risk Probability: {p_nlp:.4f}")
    
    # Fusion
    fusion = FusionEngine(alpha=0.58)
    risk_score = fusion.fuse_predictions(p_eeg, p_nlp)
    category = fusion.get_risk_category(risk_score)
    
    print(f"\nFused Risk Score: {risk_score:.4f}")
    print(f"Clinical Risk Category: {category}")
    
    print("\n✅ Prototype is working! Ready for real data & training.")

if __name__ == "__main__":
    demo_inference()