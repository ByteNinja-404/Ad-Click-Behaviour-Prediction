from preprocessing import DataPreprocessor
from visualization import DataVisualizer
from model import AdClickModel

def main():
    # Initialize components
    preprocessor = DataPreprocessor()
    visualizer = DataVisualizer()
    model = AdClickModel()
    
    # Load and preprocess data
    file_path = r'C:\Users\rishi\Desktop\DMKD Project\project\data\ad_click_dataset.csv'
    
    X_train, X_test, y_train, y_test, data = preprocessor.preprocess_data(file_path)
    
    # Create visualizations
    visualizer.create_all_visualizations(data)
    
    # Train and evaluate model
    model.train(X_train, y_train)
    results = model.evaluate(X_test, y_test)
    
    # Print results
    print(f"\nModel Accuracy: {results['accuracy']:.2f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Save model
    model.save_model()

if __name__ == "__main__":
    main()