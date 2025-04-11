import random

import torch
import torch.nn.functional as F
import pdb

class BaseRankLoss(torch.nn.Module):
    def forward(self, score, target):
        raise NotImplementedError

    def forward_per_list(self, score, target, length):
        # Split score and target into lists
        length_per_list = length.tolist()
        score_per_list = score.split(length_per_list)
        target_per_list = target.split(length_per_list)

        # Compute loss per list, giving each list equal weight (regardless of length)
        loss_per_list = [
            self(score_of_list, target_of_list)
            for score_of_list, target_of_list in zip(score_per_list, target_per_list)
        ]
        losses = torch.stack(loss_per_list)

        # Remove losses that are zero (e.g. all item labels are zero)
        losses = losses[torch.abs(losses) > 0.]
        if len(losses) == 0:
            # If all losses were removed, take the sum (which will result in a zero gradient)
            return losses.sum()

        loss = losses.mean()
        return loss


class MSELoss(BaseRankLoss):
    def forward(self, score, target):
        return torch.nn.functional.mse_loss(score, target)


class PAOrdinalLoss(BaseRankLoss):
    # See A Neural Network Approach to Ordinal Regression
    def __init__(self, higher_is_better=False):
        super().__init__()
        self.higher_is_better = higher_is_better
    def forward(self, scores, target, gamma=2.0, alpha=0.9, reduction='mean', pnl=None):
        batch_size, max_rank = scores.size()
        true_ranks = target.long()
        # Create binary matrix representation for true ranks
        true_matrix = torch.zeros((batch_size, max_rank), device=scores.device)

        for i in range(batch_size):
            rank = true_ranks[i]
            true_matrix[i, rank.item()-1:] = 1

        # Compute the binary cross-entropy loss
        BCE_loss = F.binary_cross_entropy(scores, true_matrix, reduction='none')

        # Focal loss calculation
        pt = torch.where(true_matrix == 1, scores, 1 - scores)

        focal_loss = (1 - pt) * BCE_loss

        if reduction == 'mean':
            return torch.mean(focal_loss)
        elif reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

    def forward_per_list(self, score, target, length, pnl=None):
        # Split score and target into lists
        length_per_list = length.tolist()
        score_per_list = score.split(length_per_list)
        target_per_list = target.split(length_per_list)
        if pnl is not None:
            pnl_per_list = pnl.split(length_per_list)

            # Compute loss per list, giving each list equal weight (regardless of length)
            loss_per_list = [
                self(score_of_list, target_of_list, pnl_of_list)
                for score_of_list, target_of_list, pnl_of_list in zip(score_per_list, target_per_list, pnl_per_list)
            ]
        else:
            loss_per_list = [
                self(score_of_list, target_of_list, None)
                for score_of_list, target_of_list in zip(score_per_list, target_per_list)
            ]
        losses = torch.stack(loss_per_list)

        # Remove losses that are zero (e.g. all item labels are zero)
        losses = losses[torch.abs(losses) > 0.]
        if len(losses) == 0:
            # If all losses were removed, take the sum (which will result in a zero gradient)
            return losses.sum()

        loss = losses.mean()
        return loss


class SoftmaxLoss(BaseRankLoss):
    def forward(self, score, target):
        softmax_score = torch.nn.functional.log_softmax(score, dim=-1)
        loss = -(softmax_score * target).mean()
        return loss


class PACrossEntropyLoss(BaseRankLoss):
    def __init__(self):
        super(PACrossEntropyLoss, self).__init__()

    def forward(self, score, target, pnl_weights, gamma=2.0):
        weights = torch.where(target == 1, gamma, 1.0)
        ls = torch.nn.functional.binary_cross_entropy_with_logits(score, target, weight=weights, reduction='sum')

        return ls



    def forward_per_list(self, score, target, length, pnl=None):
        # Split score and target into lists
        length_per_list = length.tolist()
        score_per_list = score.split(length_per_list)
        target_per_list = target.split(length_per_list)
        if pnl is not None:
            pnl_per_list = pnl.split(length_per_list)

            # Compute loss per list, giving each list equal weight (regardless of length)
            loss_per_list = [
                self(score_of_list, target_of_list, pnl_of_list)
                for score_of_list, target_of_list, pnl_of_list in zip(score_per_list, target_per_list, pnl_per_list)
            ]
        else:
            loss_per_list = [
                self(score_of_list, target_of_list, None)
                for score_of_list, target_of_list in zip(score_per_list, target_per_list)
            ]
        losses = torch.stack(loss_per_list)

        # Remove losses that are zero (e.g. all item labels are zero)
        losses = losses[torch.abs(losses) > 0.]
        if len(losses) == 0:
            # If all losses were removed, take the sum (which will result in a zero gradient)
            return losses.sum()

        loss = losses.mean()
        return loss


class CrossEntropyLoss(BaseRankLoss):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, score, target, gamma=2.0, alpha=0.9):
        ls = torch.nn.BCEWithLogitsLoss(reduction='sum')(score, target)
        return ls



    def forward_per_list(self, score, target, length):
        # Split score and target into lists
        length_per_list = length.tolist()
        score_per_list = score.split(length_per_list)
        target_per_list = target.split(length_per_list)

        loss_per_list = [
            self(score_of_list, target_of_list, None)
            for score_of_list, target_of_list in zip(score_per_list, target_per_list)
        ]
        losses = torch.stack(loss_per_list)

        # Remove losses that are zero (e.g. all item labels are zero)
        losses = losses[torch.abs(losses) > 0.]
        if len(losses) == 0:
            # If all losses were removed, take the sum (which will result in a zero gradient)
            return losses.sum()

        loss = losses.mean()
        return loss


class GraphBasedLoss(torch.nn.Module):
    def __init__(self, margin=1.0, epsilon=1e-8, top_k=3, train_mode=True):
        super(GraphBasedLoss, self).__init__()
        self.margin = margin
        self.epsilon = epsilon
        self.top_k = top_k
        self.train_mode = train_mode

    def log_normalize(self, tensor):
        # Apply logarithmic normalization
        # tensor_abs = torch.abs(tensor) + 1  # Add epsilon to avoid log(0)
        # tensor_log = torch.log10(tensor_abs)
        tensor_log = torch.log10(tensor+1)
        return tensor_log

    def mad_normalize(self, tensor):
        median = tensor.median()
        mad = torch.abs(tensor - median).median()  # Median Absolute Deviation
        return (tensor - median) / (mad + self.epsilon)

    def forward(self, scores, target=None, pnl=None):
        """
        Compute the Graph-Based Loss for 1-D vectors.
        :param scores: Predicted scores, a 1-D tensor of shape [num_items]
        :param pnl: Profit and Loss values for each item, a 1-D tensor of shape [num_items]
        :return: Graph-based loss
        """


        if len(scores.shape) > 1:
            scores = scores.squeeze(dim=-1)
        # sorted_pnl, sorted_indices = torch.sort(pnl, descending=True)
        top_k = min(self.top_k, len(scores))
        sorted_pnl, sorted_indices = torch.topk(pnl, k=top_k)
        sorted_target = target[sorted_indices]
        sorted_scores = scores[sorted_indices]
        # sorted_target = target[sorted_indices]
        score_diffs = sorted_scores.unsqueeze(1) - sorted_scores.unsqueeze(0)  # Shape: [num_items, num_items]
        # only keep the up triangle of the score_diffs matrix
        score_diffs_mask = torch.triu(torch.ones_like(score_diffs), diagonal=1)
        pnl_diffs = sorted_pnl.unsqueeze(1) - sorted_pnl.unsqueeze(0)
        pnl_diffs = pnl_diffs * score_diffs_mask
        normalised_pnl_diffs = self.log_normalize(pnl_diffs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(score_diffs*score_diffs_mask, score_diffs_mask, reduction='sum', weight=normalised_pnl_diffs)

        return loss / len(sorted_scores)


    def forward_per_list(self, score, target, length, pnl=None):
        # Split score and target into lists
        length_per_list = length.tolist()
        score_per_list = score.split(length_per_list)
        target_per_list = target.split(length_per_list)
        if pnl is not None:
            pnl_per_list = pnl.split(length_per_list)

            # Compute loss per list, giving each list equal weight (regardless of length)
            loss_per_list = [
                self(score_of_list, target_of_list, pnl_of_list)
                for score_of_list, target_of_list, pnl_of_list in zip(score_per_list, target_per_list, pnl_per_list)
            ]
        else:
            loss_per_list = [
                self(score_of_list, target_of_list, None)
                for score_of_list, target_of_list in zip(score_per_list, target_per_list)
            ]
        losses = torch.stack(loss_per_list)

        # Remove losses that are zero (e.g. all item labels are zero)
        losses = losses[torch.abs(losses) > 0.]
        if len(losses) == 0:
            # If all losses were removed, take the sum (which will result in a zero gradient)
            return losses.sum()

        loss = losses.mean()
        return loss


class PASoftmaxLoss(BaseRankLoss):
    def forward(self, scores, target, pnl_weights):

        if len(scores.shape) > 1:
            scores = scores.squeeze(dim=-1)

        weighted_scores = torch.where(target == 1, 0.2 * scores, 1.0 * scores)
        softmax_scores = torch.nn.functional.log_softmax(weighted_scores, dim=-1)
        softmax_loss = -(softmax_scores * target).mean()

        return softmax_loss.mean()



    def forward_per_list(self, score, target, length, pnl=None):
        # Split score and target into lists
        length_per_list = length.tolist()
        score_per_list = score.split(length_per_list)
        target_per_list = target.split(length_per_list)
        if pnl is not None:
            pnl_per_list = pnl.split(length_per_list)

            # Compute loss per list, giving each list equal weight (regardless of length)
            loss_per_list = [
                self(score_of_list, target_of_list, pnl_of_list)
                for score_of_list, target_of_list, pnl_of_list in zip(score_per_list, target_per_list, pnl_per_list)
            ]
        else:
            loss_per_list = [
                self(score_of_list, target_of_list, None)
                for score_of_list, target_of_list in zip(score_per_list, target_per_list)
            ]
        losses = torch.stack(loss_per_list)

        # Remove losses that are zero (e.g. all item labels are zero)
        losses = losses[torch.abs(losses) > 0.]
        if len(losses) == 0:
            # If all losses were removed, take the sum (which will result in a zero gradient)
            return losses.sum()

        loss = losses.mean()
        return loss


class LambdaLoss(BaseRankLoss):
    def __init__(self, eps=1e-10, padded_value_indicator=-1, weighing_scheme=None,
                 k=None, sigma=1., mu=10., reduction="sum", reduction_log="binary"):
        super().__init__()
        self.eps = eps
        self.padded_value_indicator = padded_value_indicator
        self.weighing_scheme = weighing_scheme
        self.k = k
        self.sigma = sigma
        self.mu = mu
        self.reduction = reduction
        self.reduction_log = reduction_log

    def forward(self, y_pred, y_true, pnl_weights=None):
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(0)

        device = y_pred.device
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        padded_mask = y_true == self.padded_value_indicator
        y_pred[padded_mask] = float("-inf")
        y_true[padded_mask] = float("-inf")

        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)
        # import pdb;
        # pdb.set_trace()
        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        if self.weighing_scheme != "ndcgLoss1_scheme":
            padded_pairs_mask &= (true_diffs > 0)

        ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:self.k, :self.k] = 1

        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :self.k], dim=-1).clamp(min=self.eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        if self.weighing_scheme is None:
            weights = 1.
        else:
            weights = getattr(self, self.weighing_scheme)(G, D, self.mu, true_sorted_by_preds)

        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
        weighted_probas = (torch.sigmoid(self.sigma * scores_diffs).clamp(min=self.eps) ** weights).clamp(min=self.eps)

        if self.reduction_log == "natural":
            losses = torch.log(weighted_probas)
        elif self.reduction_log == "binary":
            losses = torch.log2(weighted_probas)
        else:
            raise ValueError("Reduction logarithm base must be either 'natural' or 'binary'")

        mask = padded_pairs_mask & ndcg_at_k_mask
        if self.reduction == "sum":
            loss = -torch.sum(losses[mask])
        elif self.reduction == "mean":
            loss = -torch.mean(losses[mask])
        else:
            raise ValueError("Reduction must be either 'sum' or 'mean'")

        return loss

    def forward_per_list(self, score, target, length, pnl=None):
        length_per_list = length.tolist()
        score_per_list = score.split(length_per_list)
        target_per_list = target.split(length_per_list)
        loss_per_list = [
            self(score_of_list, target_of_list)
            for score_of_list, target_of_list in zip(score_per_list, target_per_list)
        ]
        losses = torch.stack(loss_per_list)
        losses = losses[torch.abs(losses) > 0.]
        if len(losses) == 0:
            return losses.sum()
        return losses.mean()

    # Weighing schemes
    def ndcgLoss1_scheme(self, G, D, *args):
        return (G / D)[:, :, None]

    def ndcgLoss2_scheme(self, G, D, *args):
        pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
        delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
        deltas = torch.abs(torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.) - torch.pow(torch.abs(D[0, delta_idxs]), -1.))
        deltas.diagonal().zero_()
        return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])

    def lambdaRank_scheme(self, G, D, *args):
        return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(G[:, :, None] - G[:, None, :])

    def ndcgLoss2PP_scheme(self, G, D, mu, *args):
        return mu * self.ndcgLoss2_scheme(G, D) + self.lambdaRank_scheme(G, D)

    def rankNet_scheme(self, G, D, *args):
        return 1.

    def rankNetWeightedByGTDiff_scheme(self, G, D, *args):
        return torch.abs(args[1][:, :, None] - args[1][:, None, :])

    def rankNetWeightedByGTDiffPowed_scheme(self, G, D, *args):
        return torch.abs(torch.pow(args[1][:, :, None], 2) - torch.pow(args[1][:, None, :], 2))