from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from src.core.model import load_mental_model
import uvicorn

app = FastAPI(title="NeuroGuard-XAI Inference API", version="2026.1")

# Load model (lazy)
model = None

class TextInput(BaseModel):
    text: str

@app.on_event("startup")
async def load_model():
    global model
    model = load_mental_model()

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(input: TextInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Tokenize and predict
    tokenizer = model.tokenizer
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        logits, concepts, attentions = model(inputs.input_ids, inputs.attention_mask)
    
    risk_levels = ["No Risk", "Low Risk", "Moderate Risk", "High Risk"]
    prediction = risk_levels[torch.argmax(logits, dim=1).item()]
    confidence = torch.softmax(logits, dim=1).max().item()
    
    return {
        "risk_level": prediction,
        "confidence": confidence,
        "concepts": dict(zip(model.concept_names[:len(concepts[0])], concepts[0].tolist()))
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)