# Climate Model Outcome Classification

This repository contains a project focused on classifying the outcome of climate model simulations using various machine learning models. The primary goal is to predict a binary `outcome` based on 18 input features from climate simulation data, as provided in the `climate.csv` dataset. The analysis is documented in two Jupyter Notebooks, `Climate_Classification_Project_Part1.ipynb` and `Climate_Classification_Project_Part2.ipynb`.

## Dataset

The project utilizes the `climate.csv` dataset, which contains 540 entries from climate model simulations. Each entry has 18 numerical features, representing different physical constants and correlation factors, and a single binary target variable, `outcome`. The dataset is highly imbalanced, with the positive class (1) representing approximately 91% of the data.

## Project Workflow

The project follows a standard machine learning workflow:

1.  **Data Preprocessing:** The `Study` and `Run` identifier columns are dropped. All numerical features are scaled using `StandardScaler` to ensure they have a zero mean and unit variance.
2.  **Feature Selection:** The `ExtraTreesClassifier` is used to evaluate the importance of each feature in predicting the outcome. This analysis identifies the top 10 most influential predictors, which are visualized in the notebooks.
3.  **Data Splitting:** The dataset is stratified and split into a 60% training set, a 20% validation set, and a 20% test set.
4.  **Model Training and Tuning:** Four different classification models are built, evaluated, and fine-tuned using `GridSearchCV` to find the optimal hyperparameters.
5.  **Evaluation:** Models are compared based on accuracy, precision, recall, F1-score, and their respective confusion matrices.

## Models and Performance

Four machine learning models were developed and evaluated. Each model was first trained with default parameters and then fine-tuned for optimal performance.

### Model Performance Summary

The performance of the best-tuned version of each model on the test set is summarized below.

| Model | Accuracy | F1-Score (Weighted) | Key Hyperparameters (`GridSearchCV`) |
| :--- | :---: | :---: | :--- |
| Support Vector Machine (SVM) | **96.30%** | **0.96** | `{'C': 10, 'gamma': 'scale', 'kernel': 'linear'}` |
| Neural Network (MLP) | 95.37% | 0.95 | `{'activation': 'tanh', 'alpha': 0.01, 'lr_init': 0.005}`|
| Decision Tree | 90.74% | 0.90 | `{'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 5}`|
| K-Nearest Neighbors (KNN) | 89.81% | 0.87 | `{'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'uniform'}`|

### Conclusion

The **Support Vector Machine (SVM) classifier** delivered the best overall performance. After hyperparameter tuning, the SVM model with a linear kernel achieved the highest accuracy and F1-score, making it the most effective and reliable model for this classification task. Notably, the tuned SVM model correctly identified all positive instances in the test set, resulting in zero false negatives.

## How to Run

1.  Clone the repository to your local machine:
    ```bash
    git clone https://github.com/pavi040315/climate-model-classification-project.git
    ```
2.  Navigate to the cloned directory.
3.  The notebooks are designed to be run in an environment like Google Colab or a local Jupyter Notebook server.
4.  Upload the `climate.csv` file when prompted by the notebook cell.
5.  Ensure the following Python libraries are installed:
    *   `pandas`
    *   `numpy`
    *   `scikit-learn`
    *   `matplotlib`
