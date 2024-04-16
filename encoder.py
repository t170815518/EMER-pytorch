import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Ref: Section 3.3.1 in Paper
    """
    def __init__(self, input_size, hidden_size, decoder_dim, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.forward_lstm = nn.LSTMCell(input_size,
                            hidden_size,
                            num_layers)
        self.dropout_forward = nn.Dropout()
        self.backward_lstm = nn.LSTMCell(input_size,
                                        hidden_size,
                                        num_layers)
        self.dropout_backward = nn.Dropout()

        self.fc = nn.Linear(hidden_size * 2, decoder_dim, bias=True)
        self.fh = nn.Linear(hidden_size * 2, decoder_dim, bias=True)

    def forward(self, embedding_input):
        """
        fixme: add dropout
        :param embedding_input: torch.tensor, (BATCH_SIZE, SEQ_LEN, FEATURE_DIM)
        :return: hidden_states, (initial_hiddens, initial_cells)
        hidden_states include the hidden states at every step
        """
        _, seq_len, _ = embedding_input.shape

        cell_end_states = []
        hidden_end_states = []
        # forward lstm
        hidden_states_steps = []
        hidden_states, cell_states = None, None
        for step in range(seq_len):
            if hidden_states is None or cell_states is None:
                hidden_states, cell_states = self.forward_lstm(embedding_input[:, step, :])
            else:
                hidden_states, cell_states = self.forward_lstm(embedding_input[:, step, :], (hidden_states, cell_states))
            hidden_states_steps.append(hidden_states.unsqueeze(1))
        cell_end_states.append(cell_states)
        hidden_end_states.append(hidden_states)

        # backward lstm
        hidden_states_steps_back = []
        hidden_states, cell_states = None, None
        for step in range(1, seq_len + 1):
            if hidden_states is None or cell_states is None:
                hidden_states, cell_states = self.backward_lstm(embedding_input[:, -step, :])
            else:
                hidden_states, cell_states = self.backward_lstm(embedding_input[:, -step, :], (hidden_states, cell_states))
            hidden_states_steps_back.append(hidden_states.unsqueeze(1))
        cell_end_states.append(cell_states)
        hidden_end_states.append(hidden_states)

        # ref: Equation 5, 6
        cell_end_states = torch.cat(cell_end_states, dim=-1)
        hidden_end_states = torch.cat(hidden_end_states, dim=-1)
        initial_cells = F.tanh(self.fc(cell_end_states))
        initial_hiddens = F.tanh(self.fh(hidden_end_states))

        hidden_states_steps = torch.cat(hidden_states_steps, dim=1)
        hidden_states_steps_back = torch.cat(hidden_states_steps_back, dim=1)
        hidden_states = torch.cat([hidden_states_steps, hidden_states_steps_back], dim=-1)

        return hidden_states, (initial_hiddens, initial_cells)
