import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataVisualizer:
    def __init__(self):
        # Create visuals directory if it doesn't exist
        os.makedirs('visuals', exist_ok=True)
        
    def plot_age_distribution(self, data):
        """Plot age distribution of users."""
        plt.figure(figsize=(10, 6))
        sns.histplot(data['age'], bins=30, kde=True, color='blue')
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.savefig('visuals/age_distribution.png')
        plt.close()
        
    def plot_click_rates_by_gender(self, data):
        """Plot click rates by gender."""
        plt.figure(figsize=(10, 6))
        sns.countplot(x='gender', hue='click', data=data)
        plt.title('Clicks by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.legend(title='Click', labels=['No Click', 'Click'])
        plt.savefig('visuals/clicks_by_gender.png')
        plt.close()
        
    def plot_correlation_heatmap(self, data):
        """Plot correlation heatmap"""

        numeric_data = data.select_dtypes(include=[np.number])  # Filter numeric columns
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.savefig('visuals/correlation_heatmap.png')
        plt.close()

        
    def plot_age_click_behavior(self, data):
        """Plot age distribution by click behavior."""
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data[data['click'] == 1]['age'], 
                   label='Click', fill=True, color='green')
        sns.kdeplot(data[data['click'] == 0]['age'], 
                   label='No Click', fill=True, color='red')
        plt.title('Age Distribution by Click Behavior')
        plt.xlabel('Age')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig('visuals/age_click_behavior.png')
        plt.close()
        
    def create_all_visualizations(self, data):
        """Generate all visualizations."""
        self.plot_age_distribution(data)
        self.plot_click_rates_by_gender(data)
        self.plot_correlation_heatmap(data)
        self.plot_age_click_behavior(data)