import torch
import torch.nn.functional as F
import numpy as np

class NTXentLoss(torch.nn.Module):
    def __init__(self, device, batch_size, temperature, use_cosine_similarity, reduction="sum"):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction)

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_sim
        else:
            return self._dot_sim

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_sim(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_sim(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        if zis.shape[0] != self.batch_size:
            self.batch_size = zis.shape[0]
            self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)

        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

    def forward_multi(self, zis, zjs, z_negs):
        similarity_matrix = self.similarity_function(zis, zjs)

        losses = []
        for neg_i in z_negs:
            distances = self.similarity_function(zis, neg_i)[torch.arange(zis.shape[0]), torch.arange(zis.shape[0])]
            logits = torch.cat((similarity_matrix, distances.unsqueeze(1)), dim=1)
            logits /= self.temperature
            labels = torch.arange(zis.shape[0]).to(self.device).long()
            losses.append(self.criterion(logits, labels))

        return torch.stack(losses, dim=-1).min(dim=-1)[0].mean(0)
    

class TripletMarginLoss(torch.nn.Module):
    def __init__(self, margin=2, reduce=True):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.reduce = reduce

    def distance(self, x, y):
        return ((x - y) ** 2).sum(dim=1)

    def forward(self, anc, pos, neg):
        d_pos = self.distance(anc, pos)
        d_neg = self.distance(anc, neg)
        margin = self.margin + d_pos - d_neg
        loss = torch.clamp(margin, min=0.0)
        if self.reduce:
            return loss.mean()
        return loss

