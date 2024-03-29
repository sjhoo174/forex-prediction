import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump
from const import columns_to_normalize, time_steps_predicted
from const import input_size, hidden_size, num_layers, output_size, device, seq_length, train_with_LSTM
from model import LSTMModel
from xgboost import XGBRegressor    

# Load the dataset
# Replace 'your_dataset.csv' with the path to your dataset
df = pd.read_csv('./EURUSD15.csv', delimiter='\t')
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
df[['date', 'time']] = df['date'].str.split(' ').tolist()

# Convert date and time columns to datetime dtype
df['date'] = pd.to_datetime(df['date'])
df['time'] = pd.to_datetime(df['time'], format='%H:%M').dt.time
data = df[columns_to_normalize]

train_size = 30000
train_data = data[:train_size]

# Initialize StandardScaler
normalize = True
if normalize:
    scaler = StandardScaler()
    train_data[columns_to_normalize] = scaler.fit_transform(train_data[columns_to_normalize])
    # Normalize selected columns
    dump(scaler, 'scaler.joblib')

    test_data = data[train_size:]
    test_data[columns_to_normalize] = scaler.fit_transform(test_data[columns_to_normalize])

train_data.to_pickle('./train.pkl')
test_data.to_pickle('./test.pkl')

if train_with_LSTM:
    # Convert data to PyTorch tensors
    data_tensor = torch.tensor(torch.from_numpy(train_data.values), dtype=torch.float32)

    seq_length = 30
    learning_rate = 0.001
    num_epochs = 5

    # Create input sequences
    sequences = []
    for i in range(len(data_tensor) - seq_length - time_steps_predicted):
        sequences.append((data_tensor[i:i+seq_length], data_tensor[i+seq_length+time_steps_predicted, 3]))

    # Define model, loss, and optimizer
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for idx , (seq, target) in enumerate(sequences):
            seq, target = seq.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(seq.unsqueeze(0))
            loss = criterion(output, target.unsqueeze(0))
            loss.backward()
            optimizer.step()
            if idx % 10000 == 0:
                print(idx)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the model
    torch.save(model.state_dict(), 'forex_lstm_model.pth')

else:

    # Convert data to PyTorch tensors
    data_tensor = torch.tensor(torch.from_numpy(train_data.values), dtype=torch.float32)
    # Create input sequences
    sequences = []
    for i in range(len(data_tensor) - seq_length - time_steps_predicted):
        sequences.append((data_tensor[i:i+seq_length], data_tensor[i+seq_length+time_steps_predicted, 3]))
    
    # Convert torch tensors to NumPy arrays
    X_train = [x.reshape(-1).numpy() for x, _ in sequences]
    y_train = [y.reshape(-1).numpy() for _,y in sequences]
    
    print(len(X_train))
    print(len(y_train))

    print("Begin training")
    xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=3, objective='reg:squarederror', device='cuda')

    # Train the model
    xgb_model.fit(X_train, y_train, verbose=True)

    # Save the model
    xgb_model.save_model('xgboost_forex.pt')