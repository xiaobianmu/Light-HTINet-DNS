# -- coding:UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
import math


def readD(set_matrix, num_):
    herb_d = []
    for i in range(num_):
        len_set = 1.0 / (len(set_matrix[i]) + 1)
        herb_d.append(len_set)
    return herb_d


def metrics_loss(model, test_val_loader_loss, batch_size):
    loss_sum = []
    device = next(model.parameters()).device
    for herb, gene_i, gene_j in test_val_loader_loss:
        herb, gene_i, gene_j = herb.to(device), gene_i.to(device), gene_j.to(device)
        _, _, loss = model(herb, gene_i, gene_j)
        loss_sum.append(loss.item())
    return round(np.mean(loss_sum), 4)


def hr_ndcg(indices_sort_top, index_end_i, top_k):
    hr_topK, ndcg_topK = 0, 0
    ndcg_max = [0] * top_k
    temp_max_ndcg = 0
    for i_topK in range(top_k):
        temp_max_ndcg += 1.0 / math.log(i_topK + 2)
        ndcg_max[i_topK] = temp_max_ndcg

    max_hr = top_k
    max_ndcg = ndcg_max[top_k - 1]
    if index_end_i < top_k:
        max_hr = (index_end_i) * 1.0
        max_ndcg = ndcg_max[index_end_i - 1]

    count = 0
    for gene_id in indices_sort_top:
        if gene_id < index_end_i:
            hr_topK += 1.0
            ndcg_topK += 1.0 / math.log(count + 2)
        count += 1
        if count == top_k:
            break
    return hr_topK / max_hr, ndcg_topK / max_ndcg


def largest_indices(ary, n):
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


class BPRData(data.Dataset):
    def __init__(self, train_dict=None, num_gene=0, num_ng=1, is_training=None,
                 data_set_count=0, all_rating=None, dns_k=1):
        super(BPRData, self).__init__()
        self.num_gene = num_gene
        self.train_dict = train_dict
        self.num_ng = num_ng
        self.is_training = is_training
        self.data_set_count = data_set_count
        self.all_rating = all_rating
        self.dns_k = dns_k
        self.features_fill = []

    def ng_sample(self):
        self.features_fill = []
        for herb_id in self.train_dict:
            positive_list = self.train_dict[herb_id]
            all_positive_list = self.all_rating[herb_id]
            for gene_i in positive_list:
                for t in range(self.num_ng):
                    neg_candidates = []
                    for _ in range(self.dns_k):
                        gene_j = np.random.randint(self.num_gene)
                        while gene_j in all_positive_list:
                            gene_j = np.random.randint(self.num_gene)
                        neg_candidates.append(gene_j)
                    self.features_fill.append([herb_id, gene_i, neg_candidates])

    def __len__(self):
        return self.num_ng * self.data_set_count

    def __getitem__(self, idx):
        features = self.features_fill
        return features[idx][0], features[idx][1], torch.tensor(features[idx][2], dtype=torch.long)


class EnhancedHTI(nn.Module):
    def __init__(self, herb_num, gene_num, factor_num, rel_hi, rel_ih, d_i_train, d_j_train, margin=0.5):
        super(EnhancedHTI, self).__init__()
        self.herb_gene_matrix = rel_hi[0]
        self.gene_herb_matrix = rel_ih[0]
        self.embed_herb = nn.Embedding(herb_num, factor_num)
        self.embed_gene = nn.Embedding(gene_num, factor_num)
        self.margin = margin
        for i in range(len(d_i_train)):
            d_i_train[i] = [d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i] = [d_j_train[i]]

        self.d_i_train = torch.FloatTensor(d_i_train).expand(-1, factor_num)
        self.d_j_train = torch.FloatTensor(d_j_train).expand(-1, factor_num)

        nn.init.normal_(self.embed_herb.weight, std=0.01)
        nn.init.normal_(self.embed_gene.weight, std=0.01)

    def forward(self, herb=None, gene_i=None, gene_j=None, mode='train'):
        herbs_emb = self.embed_herb.weight
        genes_emb = self.embed_gene.weight
        device = herbs_emb.device
        d_i = self.d_i_train.to(device)
        d_j = self.d_j_train.to(device)
        mat_hi = self.herb_gene_matrix.to(device)
        mat_ih = self.gene_herb_matrix.to(device)

        gcn1_herbs = torch.sparse.mm(mat_hi, genes_emb) + herbs_emb.mul(d_i)
        gcn1_genes = torch.sparse.mm(mat_ih, herbs_emb) + genes_emb.mul(d_j)
        gcn2_herbs = torch.sparse.mm(mat_hi, gcn1_genes) + gcn1_herbs.mul(d_i)
        gcn2_genes = torch.sparse.mm(mat_ih, gcn1_herbs) + gcn1_genes.mul(d_j)
        gcn3_herbs = torch.sparse.mm(mat_hi, gcn2_genes) + gcn2_herbs.mul(d_i)
        gcn3_genes = torch.sparse.mm(mat_ih, gcn2_herbs) + gcn2_genes.mul(d_j)

        all_h = torch.cat((herbs_emb, gcn1_herbs, gcn2_herbs, gcn3_herbs), -1)
        all_g = torch.cat((genes_emb, gcn1_genes, gcn2_genes, gcn3_genes), -1)

        if mode == 'predict':
            return all_h, all_g

        h_e = F.embedding(herb, all_h)
        gi_e = F.embedding(gene_i, all_g)
        gj_candidates_e = F.embedding(gene_j, all_g)
        p_i = (h_e * gi_e).sum(dim=-1)
        p_j_all = (h_e.unsqueeze(1) * gj_candidates_e).sum(dim=-1)
        p_j_hardest, hardest_indices = torch.max(p_j_all, dim=1)
        gj_hardest_e = gj_candidates_e[torch.arange(gj_candidates_e.size(0)), hardest_indices, :]
        l2_reg = 0.01 * (h_e ** 2 + gi_e ** 2 + gj_hardest_e ** 2).sum(dim=-1)
        # margin = 0.5
        loss = -F.logsigmoid(p_i - p_j_hardest - self.margin).mean() + l2_reg.mean()
        return p_i, p_j_hardest, loss


def readRelationalMatrices(training_dict, herb_num, gene_num, h_d, g_d, relation_type=0):
    indices = []
    values = []
    for h_id in training_dict:
        for g_id in training_dict[h_id]:
            indices.append([h_id, g_id])
            values.append(np.sqrt(h_d[h_id] * g_d[g_id]))
    idx = torch.LongTensor(indices).t()
    v = torch.FloatTensor(values)
    return {relation_type: torch.sparse_coo_tensor(idx, v, (herb_num, gene_num))}


def prepare_triples(training_dict):
    triples = []
    for h_id in training_dict:
        for g_id in training_dict[h_id]:
            triples.append([h_id, g_id, 0])
    return triples