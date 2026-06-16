import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

def train_and_save_models():
    print("Training EEG Model...")
    np.random.seed(42)
    X = np.random.rand(2000, 500)
    y = np.random.randint(0, 2, 2000)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_scaled, y)
    
    joblib.dump(model, 'models/eeg_model.pkl')
    joblib.dump(scaler, 'models/eeg_scaler.pkl')
    print("✅ Models trained and saved!")

if __name__ == "__main__":
    train_and_save_models()