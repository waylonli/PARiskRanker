import torch
from torch.nn.utils.rnn import pad_sequence

from ranking_model.loss import SoftmaxLoss


class RankFormer(torch.nn.Module):
    def __init__(self, input_dim,
                 tf_dim_feedforward=32,
                 tf_nhead=4,
                 tf_num_layers=4,
                 head_hidden_layers=None,
                 dropout=0.25,
                 output_embedding_mode=False
                 ):
        """
        :param input_dim: dimensionality of item features.
        :param tf_dim_feedforward: dim_feedforward in the TransformerEncoderLayer.
        :param tf_nhead: nhead in the TransformerEncoderLayer.
        :param tf_num_layers: num_layers in the TransformerEncoder that combines the TransformerEncoderLayers.
        :param head_hidden_layers: Hidden layers in score heads as list of ints. Defaults to [32].
        :param dropout: Used in score heads and TransformerEncoderLayer.
        :param list_pred_strength: Strength of the listwide loss. If 0, no listwide score head is initialized.
        """

        super().__init__()
        if head_hidden_layers is None:
            head_hidden_layers = [32]
        self.config_dict = {
            'input_dim': input_dim,
            'tf_dim_feedforward': tf_dim_feedforward,
            'tf_nhead': tf_nhead,
            'tf_num_layers': tf_num_layers,
            'head_hidden_layers': head_hidden_layers,
            'dropout': dropout,
            'output_embedding_mode': output_embedding_mode
        }
        self.input_dim = input_dim
        self.transformer = None
        self.rank_score_net = None
        self.list_emb = None
        self.list_score_net = None

        self.list_loss_fn = None
        self.list_loss_adjustment_factor = None

        encoder_layer = torch.nn.TransformerEncoderLayer(self.input_dim, nhead=tf_nhead,
                                                         dim_feedforward=tf_dim_feedforward, dropout=dropout,
                                                         activation='gelu', batch_first=True,
                                                         norm_first=True)
        # Note: the 'norm' parameter is set to 'None' here, because the TransformerEncoderLayer already computes it
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=tf_num_layers, norm=None)

        self.rank_score_net = MLP(input_dim=self.input_dim, hidden_layers=head_hidden_layers, output_dim=1,
                                  dropout=dropout)
        self.rank_loss_fn = SoftmaxLoss()
        self.output_embedding_mode = output_embedding_mode


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
        feat = torch.nn.functional.pad(feat, (0, self.input_dim - feat.shape[-1]))
        padding_mask = torch.ones((feat.shape[0], feat.shape[1]), dtype=torch.bool).to(feat.device)

        for i, list_len in enumerate(length):
            padding_mask[i, :list_len] = False

        tf_embs = self.transformer(feat, src_key_padding_mask=padding_mask)
        tf_list_emb = None
        # if self.list_emb is not None:
        #     # Extract the list embeddings
        #     tf_list_emb = tf_embs[:, 0]
        #     tf_embs = tf_embs[:, 1:]
        #     padding_mask = padding_mask[:, 1:]

            # if self.list_pred_strength > 0.:
                # Concatenate the list embeddings to the individual list element embeddings
                # tf_list_emb_expanded = tf_list_emb.unsqueeze(1).expand(-1, tf_embs.shape[1], -1)
                # tf_embs = torch.cat([tf_embs, tf_list_emb_expanded], dim=-1)

        # Only keep the non-padded list elements and concatenate all embedded list features again
        tf_embs = tf_embs[~padding_mask]

        if self.output_embedding_mode:
            return tf_embs

        rank_score = self.rank_score_net(tf_embs)
        if self.list_score_net is not None:
            list_score = self.list_score_net(tf_list_emb)
            return rank_score, list_score
        else:
            return rank_score

    def compute_loss(self, score, target, length):
        """
        :param score: See output of forward().
        :param target: Tensor of shape (N,) with the target labels for each list element.
        :param length: Tensor of shape (N,) with the length of each list.
        :return: If list_pred_strength is 0, a 0-dimensional Tensor with the ranking loss. Else, a tuple of the ranking
        loss and the listwide loss.
        """

        if isinstance(score, tuple):
            rank_score, list_score = score
        else:
            rank_score = score
            list_score = None

        rank_loss = self.rank_loss_fn.forward_per_list(rank_score, target, length)

        return rank_loss

    def get_name(self):
        return "RankFormer"

    def export_config_dict(self):
        return self.config_dict


class MLP(torch.nn.Module):
    def __init__(self, input_dim,
                 hidden_layers=None,
                 output_dim=1,
                 dropout=0.):
        super().__init__()

        net = []
        for h_dim in hidden_layers:
            net.append(torch.nn.Linear(input_dim, h_dim))
            net.append(torch.nn.ReLU())
            if dropout > 0.:
                net.append(torch.nn.Dropout(dropout))
            input_dim = h_dim
        net.append(torch.nn.Linear(input_dim, output_dim))

        self.net = torch.nn.Sequential(*net)
        self.rank_loss_fn = SoftmaxLoss()

    def forward(self, feat, *_args):
        score = self.net(feat).squeeze(dim=-1)
        return score

    def compute_loss(self, score, target, length):
        loss = self.rank_loss_fn.forward_per_list(score, target, length)
        return loss
