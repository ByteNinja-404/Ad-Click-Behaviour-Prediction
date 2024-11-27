from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

class AdClickModel:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        
    def train(self, X_train, y_train):
        """Train the Random Forest model."""
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)
        
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }
        
    def save_model(self, path='models'):
        """Save the trained model."""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, f'{path}/random_forest_model.joblib')
        
    def load_model(self, path='models'):
        """Load a trained model."""
        model_path = f'{path}/random_forest_model.joblib'
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            raise FileNotFoundError("No saved model found.")