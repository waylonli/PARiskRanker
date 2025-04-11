import torch.nn as nn
from attr import asdict
import torch
from ranking_model.transformer import make_transformer
from util import instantiate_class
from ranking_model.loss import *
from torch.nn.utils.rnn import pad_sequence

def first_arg_id(x, *y):
    return x

class PARankFormer(torch.nn.Module):
    def __init__(self, input_layer, encoder, output_layer, rank_loss_type="softmax"):
        super(PARankFormer, self).__init__()
        self.input_layer = input_layer if input_layer else torch.nn.Identity()
        self.encoder = encoder if encoder else first_arg_id
        self.output_layer = output_layer
        if rank_loss_type == "softmax":
            self.rank_loss_fn = SoftmaxLoss()
        elif rank_loss_type == "graph":
            self.rank_loss_fn = GraphBasedLoss()
        elif rank_loss_type == 'pasoftmax':
            self.rank_loss_fn = PASoftmaxLoss()
        elif rank_loss_type == 'bce':
            self.rank_loss_fn = PACrossEntropyLoss()
        else:
            raise ValueError("Unknown loss function: {}".format(rank_loss_type))

    def forward(self, feat, length):
        """
        :param feat: Tensor of shape (N, input_dim) with N the total number of list elements.
        :param length: Tensor of shape (N,) with the length of each list.
        :return: If list_pred_strength is 0, a Tensor of shape (N,) with the predicted scores for each list element.
        Else, a tuple of: 1) a Tensor of shape (N,) with the predicted scores for each list element and 2) a single
        listwide score for each list.
        """
        feat_per_list = feat.split(length.tolist())
        # Stack all lists as separate batch elements in a large tensor and add padding where needed
        feat = pad_sequence(feat_per_list, batch_first=True, padding_value=0)

        # Pad the input to the transformer to the maximum feature length
        # feat = torch.nn.functional.pad(feat, (0, self.input_dim - feat.shape[-1]))
        padding_mask = torch.ones((feat.shape[0], feat.shape[1]), dtype=torch.bool).to(feat.device)

        for i, list_len in enumerate(length):
            padding_mask[i, :list_len] = False

        # indices should be the original item ranks used in positional encoding, shape [batch_size, slate_length]
        indices = torch.arange(0, feat.shape[1]).unsqueeze(0).repeat(feat.shape[0], 1).to(feat.device)
        tf_embs = self.encoder(feat, mask=padding_mask, indices=indices)

        # Only keep the non-padded list elements and concatenate all embedded list features again
        tf_embs = tf_embs[~padding_mask]

        rank_score = self.output_layer.score(tf_embs)
        # print(rank_score)
        return rank_score

    def compute_loss(self, score, target, length, pnl=None, binary=True):
        """
        :param score: See output of forward().
        :param target: Tensor of shape (N,) with the target labels for each list element.
        :param length: Tensor of shape (N,) with the length of each list.
        :return: If list_pred_strength is 0, a 0-dimensional Tensor with the ranking loss. Else, a tuple of the ranking
        loss and the listwide loss.
        """

        rank_loss = self.rank_loss_fn.forward_per_list(score, target, length, pnl=pnl)

        return rank_loss

    def get_name(self):
        return "PA-RankFormer"

    def export_config_dict(self):
        return self.config_dict


class FCModel(torch.nn.Module):
    """
    This class represents a fully connected neural network model with given layer sizes and activation function.
    """
    def __init__(self, sizes, input_norm, activation, dropout, n_features):
        """
        :param sizes: list of layer sizes (excluding the input layer size which is given by n_features parameter)
        :param input_norm: flag indicating whether to perform layer normalization on the input
        :param activation: name of the PyTorch activation function, e.g. Sigmoid or Tanh
        :param dropout: dropout probability
        :param n_features: number of input features
        """
        super(FCModel, self).__init__()
        sizes.insert(0, n_features)
        layers = [nn.Linear(size_in, size_out) for size_in, size_out in zip(sizes[:-1], sizes[1:])]
        self.input_norm = nn.LayerNorm(n_features) if input_norm else nn.Identity()
        self.activation = nn.Identity() if activation is None else instantiate_class(
            "torch.nn.modules.activation", activation)
        self.dropout = nn.Dropout(dropout or 0.0)
        self.output_size = sizes[-1]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass through the FCModel.
        :param x: input of shape [batch_size, slate_length, self.layers[0].in_features]
        :return: output of shape [batch_size, slate_length, self.output_size]
        """
        x = self.input_norm(x)
        for layer in self.layers:
            x = self.dropout(self.activation(layer(x)))
        return x


class OutputLayer(torch.nn.Module):
    """
    This class represents an output block reducing the output dimensionality to d_output.
    """
    def __init__(self, d_model, d_output, output_activation=None):
        """
        :param d_model: dimensionality of the output layer input
        :param d_output: dimensionality of the output layer output
        :param output_activation: name of the PyTorch activation function used before scoring, e.g. Sigmoid or Tanh
        """
        super(OutputLayer, self).__init__()
        self.activation = torch.nn.Identity() if output_activation is None else instantiate_class(
            "torch.nn.modules.activation", output_activation)
        self.d_output = d_output
        self.w_1 = torch.nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Forward pass through the OutputLayer.
        :param x: input of shape [batch_size, slate_length, self.d_model]
        :return: output of shape [batch_size, slate_length, self.d_output]
        """
        # print(x.shape)
        return self.activation(self.w_1(x))

    def score(self, x):
        """
        Forward pass through the OutputLayer and item scoring by summing the individual outputs if d_output > 1.
        :param x: input of shape [batch_size, slate_length, self.d_model]
        :return: output of shape [batch_size, slate_length]
        """
        # if self.d_output > 1:
        #     return self.forward(x)
        # else:
        #     return self.forward(x).sum(-1)
        return self.forward(x)



def make_model(fc_model, transformer, post_model, n_features, rank_fn):
    """
    Helper function for instantiating LTRModel.
    :param fc_model: FCModel used as input block
    :param transformer: transformer Encoder used as encoder block
    :param post_model: parameters dict for OutputModel output block (excluding d_model)
    :param n_features: number of input features
    :return: LTR model instance
    """
    if fc_model:
        fc_model = FCModel(**fc_model, n_features=n_features)  # type: ignore
    d_model = n_features if not fc_model else fc_model.output_size
    if transformer:
        # transformer is the config dict for the transformer encoder block
        transformer = make_transformer(n_features=d_model, d_ff=transformer['d_ff'], h=transformer['h'],
                                       dropout=transformer['dropout'], positional_encoding_type=transformer['positional_encoding_type'],
                                       max_length=transformer['max_length'])
    model = PARankFormer(fc_model, transformer, OutputLayer(d_model, **post_model), rank_fn)

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    return model
