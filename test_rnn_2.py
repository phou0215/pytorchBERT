import torch
import torch.nn as nn

sentence_length = 20
word_dim = 256
hidden_size = 128
batch_size = 1

# batch_size X sentence_length X word_dim
inputs_tensor = torch.FloatTensor(batch_size, sentence_length, word_dim)
cell = nn.LSTM(input_size=word_dim, hidden_size=hidden_size,
              num_layers=2, batch_first=True, bidirectional=True, bias=True)
outputs, _status = cell(inputs_tensor)

print(outputs.shape)
print(_status[0].shape)
print(_status[1].shape)

