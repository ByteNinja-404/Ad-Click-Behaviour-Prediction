import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        
    def load_data(self, file_path):
        """Load the dataset from CSV file."""
        try:
            data = pd.read_csv(file_path)
            print("Data loaded successfully!")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
            
    def handle_missing_values(self, data):
        """Handle missing values in the dataset."""
        # Fill missing numerical values with median
        data['age'].fillna(data['age'].median(), inplace=True)
        
        # Fill missing categorical values with 'Unknown'
        categorical_columns = ['gender', 'device_type', 'ad_position', 
                             'browsing_history', 'time_of_day']
        for col in categorical_columns:
            data[col].fillna('Unknown', inplace=True)
            
        return data
        
    def encode_categorical_features(self, data):
        """Encode categorical variables using Label Encoding."""
        categorical_columns = ['gender', 'device_type', 'ad_position', 
                             'browsing_history', 'time_of_day']
                             
        for col in categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            self.label_encoders[col] = le
            
        return data
        
    def prepare_features(self, data):
        """Prepare features by dropping unnecessary columns."""
        data = data.drop(['id', 'full_name'], axis=1)
        
        # Split into features and target
        X = data.drop('click', axis=1)
        y = data['click']
        
        return X, y
        
    def split_data(self, X, y, test_size=0.3):
        """Split data into training and testing sets."""
        return train_test_split(X, y, test_size=test_size, random_state=42)
        
    def preprocess_data(self, file_path):
        """Preprocess the data: handle missing values, encode categories, and split data."""
        # Load dataset
        data = pd.read_csv(file_path)
    
        # Drop irrelevant columns
        irrelevant_columns = ['id', 'full_name']  # Add any other non-numeric columns here
        data = data.drop(columns=[col for col in irrelevant_columns if col in data.columns], errors='ignore')
    
        # Handle missing values
        data['age'] = data['age'].fillna(data['age'].median())
        for col in ['gender', 'device_type', 'ad_position', 'browsing_history', 'time_of_day']:
            data[col] = data[col].fillna('Unknown')

        # Encode categorical columns
        label_encoders = {}
        for col in ['gender', 'device_type', 'ad_position', 'browsing_history', 'time_of_day']:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

        # Ensure all features are numeric
        numeric_data = data.select_dtypes(include=[np.number])

        # Split data into features and target
        X = numeric_data.drop('click', axis=1)
        y = numeric_data['click']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        print("Data preprocessing completed successfully!")
        return X_train, X_test, y_train, y_test, numeric_data
