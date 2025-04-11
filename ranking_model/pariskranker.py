import os

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from ranking_model.ft_transformer import FTTransformer
from ranking_model.loss import *


class PARiskRanker(torch.nn.Module):
    def __init__(
            self,
            input_dim,
            tf_dim_feedforward=32,
            tf_nhead=4,
            tf_num_layers=4,
            head_hidden_layers=None,
            dropout=0.25,
            output_embedding_mode=False,
            loss_fn="graph",
            ft_embedder_checkpoint_path=None,
            ft_embedder_train_data=None
        ):

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

        encoder_layer = torch.nn.TransformerEncoderLayer(self.input_dim, nhead=tf_nhead,
                                                         dim_feedforward=tf_dim_feedforward, dropout=dropout,
                                                         activation='gelu', batch_first=True,
                                                         norm_first=True)
        # Note: the 'norm' parameter is set to 'None' here, because the TransformerEncoderLayer already computes it
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=tf_num_layers, norm=None)

        self.rank_score_net = MLP(input_dim=self.input_dim, hidden_layers=head_hidden_layers, output_dim=1,
                                  dropout=dropout)
        self.ft_embedder = torch.load(ft_embedder_checkpoint_path) if ft_embedder_checkpoint_path else self.train_ft_embedder(ft_embedder_train_data)

        # freeze the ft_embedder
        for param in self.ft_embedder.parameters():
            param.requires_grad = True

        if loss_fn == "softmax":
            self.rank_loss_fn = SoftmaxLoss()
        elif loss_fn == "graph":
            self.rank_loss_fn = GraphBasedLoss()
        elif loss_fn == 'pasoftmax':
            self.rank_loss_fn = PASoftmaxLoss()
        elif loss_fn == 'bce':
            self.rank_loss_fn = PACrossEntropyLoss()
        else:
            raise ValueError("Unknown loss function: {}".format(loss_fn))
        self.output_embedding_mode = output_embedding_mode


    def forward(self, cat_feat, conti_feat, length):
        _, feat = self.ft_embedder(cat_feat, conti_feat, return_last_hidden=True)
        feat_per_list = feat.split(length.tolist())
        # Stack all lists as separate batch elements in a large tensor and add padding where needed
        feat = pad_sequence(feat_per_list, batch_first=True, padding_value=0)

        # Pad the input to the transformer to the maximum feature length
        feat = torch.nn.functional.pad(feat, (0, self.input_dim - feat.shape[-1]))
        padding_mask = torch.ones((feat.shape[0], feat.shape[1]), dtype=torch.bool).to(feat.device)

        for i, list_len in enumerate(length):
            padding_mask[i, :list_len] = False

        tf_embs = self.transformer(feat, src_key_padding_mask=padding_mask)

        # Only keep the non-padded list elements and concatenate all embedded list features again
        tf_embs = tf_embs[~padding_mask]

        if self.output_embedding_mode:
            return tf_embs

        rank_score = self.rank_score_net(tf_embs)

        return rank_score

    def compute_loss(self, score, target, length, pnl=None, binary=True):

        rank_score = score
        if pnl is not None:
            rank_loss = self.rank_loss_fn.forward_per_list(rank_score, target, length, pnl=pnl)
        else:
            rank_loss = self.rank_loss_fn.forward_per_list(rank_score, target, length)

        return rank_loss

    def get_name(self):
        return "RankFormer"

    def export_config_dict(self):
        return self.config_dict

    def train_ft_embedder(self, train_data):
        assert train_data is not None, "Training data must be provided for training the transformer embedder"

        val_data = None
        if type(train_data) is tuple:
            train_data, val_data = train_data

        train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=1024, shuffle=True) if val_data is not None else None

        model = FTTransformer(
            categories=train_data.max_cat_classes,  # tuple containing the number of unique values within each category
            num_continuous=train_data.num_continuous,
            dim=32,  # dimension, paper set at 32
            dim_out=1,  # binary prediction, but could be anything
            depth=2,  # depth, paper recommended 6
            heads=8,  # heads, paper recommends 8
            attn_dropout=0.1,  # post-attention dropout
            ff_dropout=0.1  # feed forward dropout
        )

        optimal_val_loss = torch.inf

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()
        for epoch in tqdm(range(50), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            val_loss_sum = 0
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                cat_feat, conti_feat, target = batch['cat_feat'], batch['conti_feat'], batch[
                    'target'].unsqueeze(1)
                pred = model(cat_feat, conti_feat)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
            model.eval()

            if val_data is not None:
                with torch.no_grad():
                    for batch in val_loader:
                        cat_feat, conti_feat, target = batch['cat_feat'], batch['conti_feat'], batch[
                            'target'].unsqueeze(1)
                        pred = model(cat_feat, conti_feat)
                        loss = criterion(pred, target)
                        val_loss_sum += loss.item()
                    val_loss_sum /= len(val_loader)
                    print('epoch: {}, val loss: {:.4f}'.format(epoch, val_loss_sum))
                    if val_loss_sum < optimal_val_loss:
                        optimal_val_loss = val_loss_sum
                        stored_path = './storage/ft_embedding_model'
                        if not os.path.exists(stored_path):
                            os.makedirs(stored_path)
                        torch.save(model, './storage/ft_embedding_model/ft_embedder.pt')
            else:
                stored_path = './storage/ft_embedding_model'
                if not os.path.exists(stored_path):
                    os.makedirs(stored_path)
                torch.save(model, './storage/ft_embedding_model/ft_embedder.pt')

        return model



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
        self.activation = torch.nn.Tanh()

    def forward(self, feat, *_args):
        score = self.net(feat).squeeze(dim=-1)
        score = self.activation(score)
        return score

