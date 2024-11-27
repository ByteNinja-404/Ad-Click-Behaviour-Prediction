# Ad Click Prediction Project

## Overview
This project implements a machine learning model to predict ad clicks based on user behavior and demographics. It uses a Random Forest Classifier to analyze patterns in user interactions and make predictions about future ad engagement.

## Project Structure
```
ad-click-prediction/
├── data/
│   └── ad_click_dataset.csv
├── src/
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── model.py
│   └── main.py
├── models/
│   └── random_forest_model.joblib
├── visuals/
│   ├── age_distribution.png
│   ├── clicks_by_gender.png
│   ├── correlation_heatmap.png
│   └── age_click_behavior.png
├── requirements.txt
└── README.md
```

## Features
- Data preprocessing and cleaning
- Exploratory data analysis with visualizations
- Machine learning model for click prediction
- Model evaluation and performance metrics

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/ad-click-prediction.git
cd ad-click-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Place your dataset in the `data/` directory
2. Run the main script:
```bash
python src/main.py
```

## Data Description
The dataset includes the following features:
- age: User's age
- gender: User's gender
- device_type: Type of device used
- ad_position: Position of the advertisement
- browsing_history: User's browsing category
- time_of_day: Time when ad was shown
- click: Target variable (1 = clicked, 0 = not clicked)

## Model Performance
The Random Forest Classifier achieves:
- Accuracy: Typically around 75-80%
- Balanced performance across different user segments
- Good prediction of both clicked and non-clicked ads

## Visualizations
The project generates several visualizations:
1. Age Distribution
2. Click Rates by Gender
3. Feature Correlation Heatmap
4. Age and Click Behavior Analysis

## Contributing
1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.