# Iris Species Classification: Predicting Flower Species Using Logistic Regression

## Overview

This project demonstrates how to classify iris flower species using a logistic regression model. The dataset used contains measurements of sepal length, sepal width, petal length, and petal width for different iris species. The aim is to predict the species of an iris flower based on these measurements.

## Dataset

- **Dataset Location**: `/dataset/iris.csv`
- **Attributes**:
  - `Id`: Identifier for each sample.
  - `SepalLengthCm`: Sepal length in centimeters.
  - `SepalWidthCm`: Sepal width in centimeters.
  - `PetalLengthCm`: Petal length in centimeters.
  - `PetalWidthCm`: Petal width in centimeters.
  - `Species`: The species of the iris flower (Iris-setosa, Iris-versicolor, Iris-virginica).

## Project Structure

- **Iris Species Classification Predicting Flower Species Using Logistic Regression.ipynb**: Jupyter notebook containing the complete code for:
  - Loading the dataset
  - Data preprocessing
  - Splitting the data into training and testing sets
  - Feature scaling
  - Training a logistic regression model
  - Evaluating model performance using confusion matrix and accuracy score
  - Predicting the species of a new iris sample

- **/dataset/iris.csv**: The CSV file containing the Iris dataset.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required Python libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`

You can install the required libraries using:
```bash
pip install pandas numpy scikit-learn matplotlib
```

### Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Vishnugnath/Iris-Species-Classification-Predicting-Flower-Species-Using-Logistic-Regression.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Iris-Species-Classification-Predicting-Flower-Species-Using-Logistic-Regression
   ```
3. Open the Jupyter notebook:
   ```bash
   jupyter notebook "Iris Species Classification Predicting Flower Species Using Logistic Regression.ipynb"
   ```
4. Run the cells in the notebook to train the model and evaluate its performance.

## Model Overview

- **Algorithm**: Logistic Regression
- **Feature Scaling**: StandardScaler is used to normalize the features for better performance of the logistic regression model.
- **Evaluation Metrics**: Accuracy score, confusion matrix, and classification report are used to assess the model's performance.

## Results

The logistic regression model is trained on 80% of the data and tested on 20%. The performance metrics indicate how well the model can classify different species of iris flowers based on the input features.

## Example Prediction

You can use the trained model to predict the species of a new iris flower by providing its measurements (sepal length, sepal width, petal length, and petal width). The notebook includes a demonstration of this.

## Acknowledgements

- The Iris dataset is a classic dataset from the UCI Machine Learning Repository and is often used for testing machine learning algorithms.
- This project uses the Iris dataset to illustrate how logistic regression can be used for classification tasks.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
