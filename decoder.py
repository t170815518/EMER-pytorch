import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, emsize, hidden_size, token_num, lstm_input_size, num_layers=1, dropout=0.1):
        super().__init__()

        self.word_embedding = nn.Embedding(token_num, emsize)
        self.lstm_layer = nn.LSTMCell(input_size=lstm_input_size,
                                  hidden_size=hidden_size,)

        self.U = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.W_h = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)
        self.bias = nn.Parameter(torch.zeros(1, hidden_size * 2, dtype=float), requires_grad=True)
        self.V = nn.Parameter(torch.ones(1, hidden_size * 2, dtype=float), requires_grad=True)
        self.W_c = nn.Linear(hidden_size*2, hidden_size*2, bias=False)

        # final layer
        self.Wo = nn.Linear(hidden_size * 3, token_num, bias=True)

    def forward(self, seq, hidden, initial_values, encoder_hiddens, is_return_more: bool = False):
        word_embeddings = self.word_embedding(seq)

        batch_size, seq_len, _ = word_embeddings.shape
        converge = None
        converge_loss = 0

        hidden_states, cell_states = initial_values
        output_hiddens = []
        output_states = []

        # set the initial h*
        e = self.W_h(encoder_hiddens)
        e = e + self.U(cell_states).unsqueeze(1).repeat(1, encoder_hiddens.shape[1], 1)
        e = e + self.bias
        e = self.V * F.tanh(e)
        a = F.softmax(e, dim=1)
        hidden_states_prime = encoder_hiddens * a   # -> (batch_len, 7, 32)
        hidden_states_prime = torch.sum(hidden_states_prime, dim=1) # -> (batch_len, 32)
        for step in range(seq_len):
            embeddings_step_ = word_embeddings[:, step, : ] # -> (batch_len, 16)
            inputs = torch.cat([embeddings_step_, hidden_states_prime, hidden], dim=1)
            hidden_states, cell_states = self.lstm_layer(inputs.float(), (hidden_states, cell_states))

            # update hidden states with converge mechansim
            e = self.W_h(encoder_hiddens)
            e = e + self.U(cell_states).unsqueeze(1).repeat(1, encoder_hiddens.shape[1], 1)
            e = e + self.bias
            if converge is not None:
                e = e +  self.W_c(converge.float())
            e = self.V * F.tanh(e)
            a = F.softmax(e, dim=1)    # fixme: dim
            hidden_states_prime = torch.sum(encoder_hiddens * a, dim=1)

            # update converge
            if converge is None:
                converge = a
                converge_loss = converge_loss + a
            else:
                converge = converge + a
                converge_loss = converge_loss + torch.where(converge < a, converge, a)

            output_hiddens.append(hidden_states_prime.unsqueeze(1))
            output_states.append(cell_states.unsqueeze(1))

        # calculate h_prime final
        e = self.W_h(encoder_hiddens)
        e = e + self.U(cell_states).unsqueeze(1).repeat(1, encoder_hiddens.shape[1], 1)
        e = e + self.bias
        e = e + self.W_c(converge.float())
        e = self.V * F.tanh(e)
        a = F.softmax(e, dim=1)
        hidden_states_prime = encoder_hiddens * a   # -> (batch_len, 7, 32)
        hidden_states_prime = torch.sum(hidden_states_prime, dim=1) # -> (batch_len, 32)
        output_hiddens.append(hidden_states_prime.unsqueeze(1))

        output_hiddens = torch.cat(output_hiddens[1:], dim=1)
        output_states = torch.cat(output_states, dim=1)
        # final layer
        final_layer_input = torch.cat([output_states, output_hiddens], dim=-1).float()
        output = F.log_softmax(self.Wo(final_layer_input), dim=-1)
        if is_return_more:
            return output, converge_loss, hidden_states, cell_states
        else:
            return output, converge_loss
