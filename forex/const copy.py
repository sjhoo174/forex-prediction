columns_to_normalize = ['open', 'high', 'low', 'close']

import torch


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyperparameters
input_size = 4
hidden_size = 64
num_layers = 2
output_size = 1

seq_length = 3000 # 30 days
time_steps_predicted = 300 # 3 days

train_with_LSTM = False
load_with_LSTM = False
