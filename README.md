# Ad Click Prediction Project

## Overview
This project focuses on predicting whether a user will click on an advertisement based on their behavior, demographics, and interaction history. Using machine learning, specifically a Random Forest Classifier, the project identifies patterns in user engagement to optimize ad targeting strategies.

By preprocessing the data, visualizing trends through exploratory data analysis (EDA), and training a predictive model, the project provides actionable insights and robust performance metrics. Key visualizations like age distributions, click rates by gender, and correlation heatmaps offer a deeper understanding of user behavior.

The final model achieves a respectable accuracy of 71% and highlights key predictors of ad click behavior, making it a valuable tool for improving ad placements and engagement strategies.

## Project Structure
```
ad-click-prediction/
├── data/
│   └── ad_click_dataset.csv
├── src/
|   ├── models/
│   ├── visuals/
|   │    ├── age_distribution.png
│   |    ├── clicks_by_gender.png
│   |    ├── correlation_heatmap.png
│   |    └── age_click_behavior.png
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── model.py
│   └── main.py
├── .gitignore
├── LICENCE
├── README.md
└── requirements.txt
```

## Features

### Data Preprocessing
- Handles missing values:
  - Fills missing numeric values (e.g., `age`) with the median.
  - Fills missing categorical values (e.g., `gender`, `device_type`) with "Unknown."
- Encodes categorical variables using `LabelEncoder` for machine learning compatibility.
- Filters out non-numeric and irrelevant columns (e.g., `id`, `full_name`).

### Exploratory Data Analysis (EDA)
- Generates insightful visualizations to uncover data patterns:
  1. **Age Distribution**: Shows the frequency of users across age groups.
  2. **Clicks by Gender**: Compares ad click behavior between genders.
  3. **Correlation Heatmap**: Highlights relationships between numerical features.
  4. **Click Behavior by Age**: Analyzes click trends based on age groups.

### Machine Learning Model
- Implements a **Random Forest Classifier** to predict user click behavior.
- Splits the dataset into training and testing subsets (70% train, 30% test).
- Evaluates the model using metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score

### Model Persistence
- Saves the trained model as `random_forest_model.joblib` for future predictions.
- Supports reloading the saved model for quick usage without retraining.

### Easy-to-Run Workflow
- Combines preprocessing, visualization, and model training into a single pipeline via `main.py`.
- Automatically saves all visualizations in the `visuals/` directory.


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

| Feature            | Description                                       |
|--------------------|---------------------------------------------------|
| **age**            | User's age                                        |
| **gender**         | User's gender                                     |
| **device_type**    | Type of device used (e.g., Mobile, Desktop)       |
| **ad_position**    | Position of the advertisement on the webpage      |
| **browsing_history** | Categorized browsing history                     |
| **time_of_day**    | Time of day when the ad was displayed             |
| **click**          | Target variable (1 = clicked, 0 = not clicked)    |

Each row in the dataset represents one user interaction with an advertisement. The goal is to predict whether the user will click the ad based on the given features.

## Model Performance

### Evaluation Metrics
The trained **Random Forest Classifier** achieves the following results on the test data:

| Metric                     | Value   |
|----------------------------|---------|
| **Accuracy**                | 0.71    |
| **Precision (Class 0)**     | 0.62    |
| **Precision (Class 1)**     | 0.73    |
| **Recall (Class 0)**        | 0.43    |
| **Recall (Class 1)**        | 0.86    |
| **F1-Score (Class 0)**      | 0.51    |
| **F1-Score (Class 1)**      | 0.79    |

#### Classification Report:
```bash
          precision    recall  f1-score   support

       0       0.62      0.43      0.51      1055
       1       0.73      0.86      0.79      1945

accuracy                           0.71      3000
```
macro avg 0.68 0.64 0.65 3000 weighted avg 0.69 0.71 0.69 3000

## Visualizations

### Age Distribution
This chart shows the distribution of users based on age.

![Age Distribution](visuals/age_distribution.png)

### Click Rates by Gender
This bar chart compares the click-through rate (CTR) between male and female users.

![Clicks by Gender](visuals/clicks_by_gender.png)

### Feature Correlation Heatmap
This heatmap visualizes the correlations between numerical features in the dataset.

![Correlation Heatmap](visuals/correlation_heatmap.png)

### Age and Click Behavior
This chart compares the age distribution of users who clicked and those who did not click on ads.

![Age Click Behavior](visuals/age_click_behavior.png)



## Contributing
1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.
