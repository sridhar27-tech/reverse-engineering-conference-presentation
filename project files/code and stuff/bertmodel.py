def create_bert_model(num_labels=4):
    print("✅ NLP Module: BERT-base placeholder ready")
    print("   (In production: Use transformers.AutoModelForSequenceClassification)")
    return "tokenizer-placeholder", None

def predict_risk(text: str):
    """Placeholder for real BERT inference"""
    print(f"Analyzing: {text[:80]}...")
    # Return probability of high suicide risk
    return 0.72