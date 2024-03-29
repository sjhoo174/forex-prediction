import pickle
import matplotlib.pyplot as plt
import pandas as pd
from const import columns_to_normalize
import torch
from model import LSTMModel
from const import input_size, hidden_size, num_layers, output_size, device, seq_length, time_steps_predicted, load_with_LSTM
from xgboost import XGBRegressor    

train_df = pd.DataFrame()

with open(r"./train.pkl", "rb") as input_file:
    train_df = pickle.load(input_file)

train_y = train_df["close"].tolist()
train_x = [i for i in range(len(train_y))]

test_df = pd.DataFrame()

with open(r"./test.pkl", "rb") as input_file:
    test_df = pickle.load(input_file)

total_df = pd.concat([train_df, test_df])

test_y = test_df["close"].tolist()
test_df = pd.concat([train_df[-seq_length:],test_df])

test_x = [i for i in range(len(train_x),len(train_x) + len(test_y))]

if load_with_LSTM:
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    model_weights_path = 'forex_lstm_model.pth'
    model.load_state_dict(torch.load(model_weights_path))

    data_tensor = torch.tensor(torch.from_numpy(test_df.values), dtype=torch.float32)

    sequences = []
    for i in range(len(data_tensor) - seq_length):
        sequences.append(data_tensor[i:i+seq_length])
    
else:
    # Load the XGBoost model
    model = XGBRegressor()
    model.load_model('xgboost_forex.pt')

    data_tensor = torch.tensor(torch.from_numpy(test_df.values), dtype=torch.float32)
    sequences = []
    for i in range(len(data_tensor) - seq_length):
        sequences.append(data_tensor[i:i+seq_length])
    
    sequences = [x.reshape((1,-1)).numpy() for x in sequences]


predictions = []
predict_interval = 100

total_profit = 0
total_wins = 0
total_loss = 0
total_trades = 0
take_profit_margin = 0.3
stop_loss_margin = -0.15
trigger_threshold = 0.6
profits = []

for idx in list(range(0, len(sequences), predict_interval)) + [len(sequences)-1]:
    seq = sequences[idx]
    x1 = idx+len(train_x)
    x2 = x1 + time_steps_predicted

    if load_with_LSTM:
        seq = seq.to(device)
        y2 = model(seq.unsqueeze(0)).item()
        y1 = sequences[idx][-1][-1].item()
    else:
        # Make predictions
        y2 = model.predict(seq)[0]
        y1 = sequences[idx][-1][-1]

    traded = False

    if x2 < len(total_df):
        # buy or sell or hold your horses
        triggered = abs(y2-y1) > trigger_threshold
        if triggered:
            action = 'BUY'
            if y2-y1 < 0:
                action = 'SELL'
            
            if action == 'BUY':
                for k in range(x1, len(total_df)):
                    price = total_df.iloc[k]['close']
                    profit = price-y1
                    if take_profit_margin <= profit or stop_loss_margin >= profit:
                        # we either take profit or loss
                        if profit >= 0:
                            total_wins += profit
                        else:
                            total_loss += profit
                        total_profit += profit
                        total_trades += 1
                        profits.append(profit)
                        traded = True
                        break
            
            if action == "SELL":
                for k in range(x1, len(total_df)):
                    price = total_df.iloc[k]['close']
                    profit = y1 - price

                    if take_profit_margin <= profit or stop_loss_margin >= profit:
                        # we either take profit or loss
                        if profit >= 0:
                            total_wins += profit
                        else:
                            total_loss += profit
                        total_profit += profit     
                        total_trades += 1  
                        profits.append(profit) 
                        traded = True  
                        break

    predictions.append((x1,x2,y1,y2,traded))

# plt.plot(train_x, train_y, label='train', color='blue')
# scale
scale_factor = len(total_df)
test_x = [i/scale_factor for i in test_x]
plt.plot(test_x, test_y, label='test', color='blue')

for prediction in predictions:
    x1,x2,y1,y2,traded = prediction
    print(prediction)
    x1,x2 = x1/scale_factor, x2/scale_factor
    dx = x2 - x1
    dy = y2 - y1
    
    color = 'red' if not traded else 'green'
    plt.arrow(x1,y1,dx,dy, label='predicted', color=color, length_includes_head=False, head_width=0.005, head_length=0.01)


plt.text(0.7,-1, f'Total profit: {total_profit}', fontsize=8, color='black')
plt.text(0.7,-1.2, f'Total wins: {total_wins}', fontsize=8, color='black')
plt.text(0.7,-1.4, f'Total loss: {total_loss}', fontsize=8, color='black')
plt.text(0.7,-1.6, f'Total trades: {total_trades}', fontsize=8, color='black')
plt.text(0.7,-1.8, f'Stop loss margin: {stop_loss_margin}', fontsize=8, color='black')
plt.text(0.7,-2, f'Take profit margin: {take_profit_margin}', fontsize=8, color='black')


# plt.legend()
plt.savefig('plot2.png')  # Save the plot as 'plot.png

plt.clf()
plt.hist(profits, bins=30, edgecolor='black')  # Adjust the number of bins as needed


plt.title('Profit frequency')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig('hist.png')  

print(f'Total profit: {total_profit}')
print(f'Total wins: {total_wins}')
print(f'Total loss: {total_loss}')
print(f'Total trades: {total_trades}')




