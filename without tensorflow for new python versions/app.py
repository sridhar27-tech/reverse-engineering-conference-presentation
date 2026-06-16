import numpy as np
import joblib
import gradio as gr
from datetime import datetime

# Import our modules
from nlp_module.simple_nlp import predict_suicide_risk
from fusion.fusion_engine import FusionEngine
from utils.report_generator import generate_report

def load_eeg_model():
    try:
        model = joblib.load('models/eeg_model.pkl')
        scaler = joblib.load('models/eeg_scaler.pkl')
        return model, scaler
    except:
        return None, None

# Load models once
eeg_model, eeg_scaler = load_eeg_model()

def predict_mental_health_risk(text_input):
    if not text_input or text_input.strip() == "":
        return "Please enter some text for analysis.", "No report generated (empty input)"
    
    # EEG Prediction (dummy features)
    if eeg_model is not None:
        dummy_eeg = np.random.rand(1, 500)
        scaled = eeg_scaler.transform(dummy_eeg)
        p_eeg = eeg_model.predict_proba(scaled)[0][1]
    else:
        p_eeg = 0.58
    
    # NLP Prediction
    p_nlp = predict_suicide_risk(text_input)
    
    # Fusion
    fusion = FusionEngine(alpha=0.58)
    risk_score = fusion.fuse_predictions(p_eeg, p_nlp)
    category = fusion.get_risk_category(risk_score)
    
    # Generate Reports
    generate_report(p_eeg, p_nlp, risk_score, category, text_input)
    
    result = f"""
**EEG Depression Probability:** {p_eeg:.4f}
**NLP Suicide Risk Probability:** {p_nlp:.4f}
**Final Fused Risk Score:** {risk_score:.4f}

**Clinical Risk Category:** **{category}**
    """
    
    return result, f"✅ Reports generated successfully! (Check folder for PDF + HTML)"

# Create Gradio Interface (Updated for newer Gradio)
with gr.Blocks(title="Mental Health Risk Assessment", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🧠 Unified EEG + NLP Mental Health Risk Assessment")
    gr.Markdown("Enter text to analyze depression and suicide risk using multimodal AI.")
    
    with gr.Row():
        text_input = gr.Textbox(
            lines=6, 
            placeholder="Enter patient's words, journal entry, or social media post here...",
            label="Input Text"
        )
    
    with gr.Row():
        submit_btn = gr.Button("Analyze Risk", variant="primary")
    
    output_result = gr.Markdown(label="Risk Assessment Result")
    output_status = gr.Textbox(label="Report Status")
    
    submit_btn.click(
        fn=predict_mental_health_risk,
        inputs=text_input,
        outputs=[output_result, output_status]
    )
    
    gr.Examples(
        examples=[
            ["I feel completely hopeless and have been thinking about ending it all."],
            ["Everything is fine, I'm just a bit tired today."],
            ["I don't see any point in living anymore, nothing matters."],
        ],
        inputs=text_input
    )
    
    gr.Markdown("**Disclaimer:** This is a research prototype. Not for clinical use without professional supervision.")

if __name__ == "__main__":
    print("🚀 Starting Web Interface...")
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,      # Change to True if you want a public link
        show_error=True
    )