import torch
import torch.nn as nn
from .rating_prediction import RatingPrediction
from .encoder import Encoder
from .decoder import Decoder


class EMER(nn.Module):
    def __init__(self, config):
        """
        self.encoder = NRTEncoder(config['user_num'], config['item_num'], config['embedding_size'],
                                  config['hidden_size'], config['nlayers'], config['max_rating'], config['min_rating'])
        """
        super().__init__()

        # rating prediction
        self.uidW = nn.Embedding(config['user_num'], config['embedding_size'],)
        self.iidW = nn.Embedding(config['item_num'], config['embedding_size'],)
        self.rating_predictor = RatingPrediction(input_dim=config['embedding_size'] * 2,
                                                 hidden_size=config['embedding_size'] * 3,
                                                 hidden_layer_num=2)

        # sentence generation
        self.uidW_generation = nn.Embedding(config['user_num'], config['embedding_size'], )
        self.iidW_generation = nn.Embedding(config['item_num'], config['embedding_size'], )
        self.titleW = nn.Embedding(config['token_num'], config['embedding_size'])
        self.encoder = Encoder(input_size=config['embedding_size'],
                               hidden_size=config['embedding_size'],
                               decoder_dim=config['embedding_size'])    # fixme: confirm the decoder's dimension
        self.decoder = Decoder(config['embedding_size'],
                               config['embedding_size'], config['token_num'], lstm_input_size=192)   # fixme: har

    def forward(self, user, item, seq, title, is_return_more: bool = False):
        """
        :param title: the item title
        """
        # rating prediction
        user_emb = self.uidW(user)
        item_emb = self.iidW(item)
        rating, features = self.rating_predictor(user_emb, item_emb)

        # sentence generation
        title_emb = self.titleW(title)  # title embedding
        user_emb_generation = self.uidW_generation(user).unsqueeze(1)
        item_emb_generation = self.iidW_generation(item).unsqueeze(1)
        encoder_input = torch.concat([user_emb_generation, item_emb_generation, title_emb], axis = 1)
        encoder_hiddens, (initial_h, initial_c) = self.encoder(encoder_input)
        if is_return_more:
            log_word_prob, converge_loss, hidden_states, cell_states = self.decoder(
                    seq, features, (initial_h, initial_c), encoder_hiddens, is_return_more=is_return_more)
            return rating, log_word_prob, converge_loss, features, encoder_hiddens, hidden_states, cell_states
        else:
            log_word_prob, converge_loss = self.decoder(seq, features, (initial_h, initial_c), encoder_hiddens,
                                                        is_return_more=is_return_more)
            return rating, log_word_prob, converge_loss


if __name__ == '__main__':
    # unit test of emer
    config_dict = {'user_num': 128, 'item_num': 128,
                   'embedding_size': 16, 'token_num': 100, }
    model = EMER(config_dict)
    user = torch.tensor([0, 1, 2])
    item = torch.tensor([0, 2, 1])
    seq = torch.tensor([[0, 1, 1, 5, 4],
                       [0, 1, 1, 5, 4],
                       [0, 1, 1, 5, 4]])
    title = torch.tensor([[0, 1, 1, 5, 4],
                       [0, 1, 1, 5, 4],
                       [0, 1, 1, 5, 4]])
    output = model(user, item, seq, title)
    print(output)