# neural-network-challenge-1

# Student Loan Risk Prediction with Deep Learning

This project involves building a neural network model to predict student loan repayment success. The model uses various student and loan-related features to classify whether a student will likely repay their loan.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Steps](#project-steps)
  - [Data Preparation](#data-preparation)
  - [Model Building](#model-building)
  - [Model Evaluation](#model-evaluation)
  - [Saving and Loading the Model](#saving-and-loading-the-model)
  - [Prediction](#prediction)
- [Results](#results)
- [Discussion](#discussion)

## Installation

1. Install the required packages:
pandas
tensorflow
Dense from tensorflow.keras.layers
Sequential from tensorflow.keras.models
StandardScaler from sklearn.preprocessing
train_test_split from sklearn.model_selection
classification_report from sklearn.metrics
Path from pathlib


## Usage

To run the project, follow the steps below:

1. Prepare the data:
    - Download the dataset from [here](https://static.bc-edx.com/ai/ail-v-1-0/m18/lms/datasets/student-loans.csv)
    - Place the `student-loans.csv` file in the `Resources` folder

2. Run the script:
    ```sh
    python student_loan_risk.py
    ```

## Project Steps

### Data Preparation

1. **Read the Data**: Load the `student-loans.csv` file into a Pandas DataFrame.
    ```python
    loans_df = pd.read_csv(file_path)
    loans_df.head()
    ```

2. **Feature and Target Selection**: Define features (X) and target (y) datasets.
    ```python
    y = loans_df["credit_ranking"]
    X = loans_df.drop(columns="credit_ranking")
    ```

3. **Data Splitting**: Split the data into training and testing sets.
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    ```

4. **Feature Scaling**: Scale the feature data using `StandardScaler`.
    ```python
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    ```

### Model Building

1. **Define the Model**: Create a sequential neural network model with TensorFlow.
    ```python
    nn_model = tf.keras.models.Sequential()
    nn_model.add(tf.keras.layers.Dense(units=6, activation="relu", input_dim=11))
    nn_model.add(tf.keras.layers.Dense(units=3, activation="relu"))
    nn_model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
    ```

2. **Compile the Model**: Compile the model using binary cross-entropy loss and the Adam optimizer.
    ```python
    nn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    ```

3. **Train the Model**: Fit the model with the training data.
    ```python
    fit_model = nn_model.fit(X_train_scaled, y_train, epochs=50)
    ```

### Model Evaluation

1. **Evaluate the Model**: Evaluate the model using test data.
    ```python
    model_loss, model_accuracy = nn_model.evaluate(X_test_scaled, y_test, verbose=2)
    print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
    ```

### Saving and Loading the Model

1. **Save the Model**: Save the model to a Keras file.
    ```python
    file_path = Path("student_loans.keras")
    nn_model.save(file_path)
    ```

2. **Load the Model**: Load the model from the saved file.
    ```python
    nn_imported = tf.keras.models.load_model(file_path)
    ```

### Prediction

1. **Make Predictions**: Use the model to make predictions on the test data.
    ```python
    predictions = nn_model.predict(X_test_scaled, verbose=2)
    ```

2. **Save Predictions**: Save the predictions to a DataFrame and round them to binary results.
    ```python
    predictions_df = pd.DataFrame(columns=["predictions"], data=predictions)
    predictions_df["predictions"] = round(predictions_df["predictions"], 0)
    ```

3. **Classification Report**: Display a classification report with the test data and predictions.
    ```python
    print(classification_report(y_test, predictions_df["predictions"].values))
    ```

## Results

The model achieved an accuracy of approximately 73% on the test data, with balanced precision and recall metrics.

## Discussion

### Creating a Recommendation System for Student Loans

1. **Data Collection**: Collect data on credit scores, financial aid, employment prospects, loan details, and academic performance to recommend suitable loan options.
2. **Filtering Method**: Use content-based filtering to match student attributes with loan features for personalized recommendations.
3. **Challenges**: Address data privacy and security concerns and mitigate bias to ensure fair and secure recommendations.

## Sources
Activites from Module 18 and Xpert Learning Assistant for README formatting suggestions.

### Conclusion

This project demonstrates the application of deep learning for predicting student loan repayment success and provides a foundation for developing a recommendation system for student loans.

