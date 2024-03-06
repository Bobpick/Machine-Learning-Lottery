import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import random

def calculate_arithmetic_complexity(numbers):
    unique_diffs = set()
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            unique_diffs.add(abs(numbers[i] - numbers[j]))
    return len(unique_diffs) - (len(numbers) - 1)

# Repeat the entire process 10 times
for _ in range(10):
    # Load the dataset, skipping rows containing field names
    df = pd.read_csv("C:/Users/Bob/Documents/Thunt/Thunt100.csv", header=None, skiprows=1)

    # Define window length and number of features
    window_length = 7
    number_of_features = 6  # Assuming there are 5 unique numbers in each row

    # Normalize the data
    scaler = StandardScaler()
    scaled_train_samples = scaler.fit_transform(df.iloc[:, 1:])  # Exclude the date column
    x_train, y_train = [], []
    for i in range(len(df) - window_length):
        x_train.append(scaled_train_samples[i : i+window_length])
        y_train.append(scaled_train_samples[i+window_length])
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Define the model architecture
    model = Sequential([
        Bidirectional(LSTM(240, input_shape=(window_length, number_of_features), return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(240, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(240, return_sequences=True)),
        Bidirectional(LSTM(240, return_sequences=False)),  # Last LSTM layer without return_sequences
        Dropout(0.2),
        Dense(37, kernel_regularizer=regularizers.l2(0.01)),  # L2 regularization with factor 0.01
        Dense(5)  # 6 for date and 5 numbers
    ])

    # Compile the model
    optimizer = Adam(learning_rate=0.00008)
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, batch_size=100, epochs=2800, verbose=2)

    # Prepare test data
    df_test = pd.read_csv("C:/Users/Bob/Documents/Thunt/Thunt100.csv", header=None)

    # Exclude the date column and scale
    x_test = scaler.transform(df_test.iloc[:, 1:].values[-window_length:])

    # Calculate the batch size dynamically based on the shape of x_test
    batch_size = x_test.shape[0]  # First dimension is the batch size

    # Repeat x_test to create a batch with the calculated batch size
    x_test = np.tile(x_test, (batch_size, 1, 1))

    # Generate 5 sets of predictions
    y_test_pred_sets = []
    for _ in range(1):
        # Generate prediction
        y_test_pred = model.predict(x_test)
        # Inverse transform and convert to integers
        y_test_pred_inv = scaler.inverse_transform(y_test_pred).astype(int).reshape(-1)

        # Add unique predictions to the set
        unique_pred_set = set(y_test_pred_inv)  # Combine and deduplicate
        while len(unique_pred_set) < 5:  # Continue generating predictions until you have 5 unique numbers
            # Generate prediction
            y_test_pred = model.predict(x_test)
            # Inverse transform and convert to integers
            y_test_pred_inv = scaler.inverse_transform(y_test_pred).astype(int).reshape(-1)

            # Add unique predictions to the set
            unique_pred_set = unique_pred_set.union(set(y_test_pred_inv))  # Combine and deduplicate

        # Sort the predictions and convert the set to a tuple
        sorted_pred_set = tuple(sorted(unique_pred_set))
        # Append the sorted tuple to the list of prediction sets
        y_test_pred_sets.append(sorted_pred_set)


    # Write the predicted numbers to a file
    with open("predicted_numbers.txt", "a") as file:
        # Write each set of 5 unique numbers to the file
        for pred_set in y_test_pred_sets:
            file.write(','.join(map(str, pred_set)))
            # Calculate the sum of each line and AC
            numbers = [int(num) for num in pred_set]
            line_sum = sum(numbers)
            ac = calculate_arithmetic_complexity(numbers)
            file.write(f"   Sum: {line_sum}, AC: {ac}\n")

print("Predictions for test data and corresponding statistics have been written to predicted_numbers.txt.")
