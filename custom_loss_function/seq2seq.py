import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim , dropout_prob, batch_size, ):
        super(EncoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.batch_size = batch_size

        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

    def forward(self, input, hidden):  # input [batch_size, length T, dimensionality d]
        #print('encoder input:', input.shape)
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def init_hidden(self, device):
        # [num_layers*num_directions,batch,hidden_size]
        return torch.zeros(self.layer_dim,  self.batch_size, self.hidden_dim, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim,
                 fc_units,
                 dropout_prob,
                 output_dim):
        super(DecoderRNN, self).__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_dim, fc_units)
        self.out = nn.Linear(fc_units, output_dim)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = F.relu( self.fc(output))
        output = self.out(output)
        return output, hidden


class Net_GRU(nn.Module):
    def __init__(self, encoder, decoder, target_length, input_dim, device):
        super(Net_GRU, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_length = target_length
        self.device = device
        self.input_dim = input_dim

    def forward(self, x):
        input_length = x.shape[1]
        encoder_hidden = self.encoder.init_hidden(self.device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(x[:, ei:ei + 1, :], encoder_hidden)

        decoder_input = x[:, -1, :].unsqueeze(1)  # first decoder input= last element of input sequence
        decoder_hidden = encoder_hidden

        outputs = torch.zeros([x.shape[0], self.input_dim, self.target_length]).to(self.device)
        for di in range(self.target_length):
            decoder_output, (decoder_hidden) = self.decoder(decoder_input, (decoder_hidden))
            decoder_input = decoder_output
            outputs[:, di:di + 1, :] = decoder_output
        return outputs