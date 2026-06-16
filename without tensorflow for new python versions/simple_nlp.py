def predict_suicide_risk(text: str) -> float:
    """Simple NLP risk predictor"""
    text_lower = text.lower()
    risk_keywords = ["hopeless", "suicide", "kill myself", "end it", "die", "worthless", "no point"]
    
    score = 0.35
    for kw in risk_keywords:
        if kw in text_lower:
            score += 0.13
    return min(max(score, 0.1), 0.95)