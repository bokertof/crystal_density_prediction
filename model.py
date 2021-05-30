import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class RNN(nn.Module):
    def __init__(self, input_size, embedd_size, hidden_size, num_layers, bidirect = False):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedd_size)

        self.RNN = nn.GRU(embedd_size, hidden_size, num_layers, bidirectional = bidirect)

        size = hidden_size*(1+bidirect)

        self.fc = nn.Linear(size, 1)

        
    def forward(self, x):
        
        embedd = self.embedding(x.data)

        temp = PackedSequence(embedd,
                            x.batch_sizes,
                            sorted_indices=x.sorted_indices.to(device),
                            unsorted_indices=x.unsorted_indices.to(device))

        _, h_gru = self.RNN(temp)
        
        h_gru = h_gru[-(1 + int(self.RNN.bidirectional)):]
        h = torch.cat(h_gru.split(1), dim=-1).squeeze(0)

        density = self.fc(h)
        
        return density
