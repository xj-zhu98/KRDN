import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum, scatter_softmax
import time
import math
import sklearn


class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """

    def __init__(self, n_users, n_items, gamma, max_iter):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.gamma = gamma
        self.max_iter = max_iter
        self.W1 = nn.Linear(64, 64)
        self.W2 = nn.Linear(64, 64)
        self.activation = nn.LeakyReLU()

    def KG_forward(self, entity_emb, edge_index, edge_type,
                   relation_weight):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index

        idx = edge_index < self.n_items
        edge_idx_cross = idx[0] ^ idx[1]
        edge_idx_same = ~edge_idx_cross

        # triplets in cross level
        edge_relation_emb = relation_weight[
            edge_type[edge_idx_cross]]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail[edge_idx_cross]] * edge_relation_emb  # [-1, channel]
        entity_agg_1 = scatter_mean(src=neigh_relation_emb, index=head[edge_idx_cross], dim_size=n_entities, dim=0)
        entity_agg_1 = self.activation(self.W1(entity_agg_1)) / 2

        # triplets in same level
        edge_relation_emb = relation_weight[edge_type[edge_idx_same]]
        neigh_relation_emb = entity_emb[tail[edge_idx_same]] + edge_relation_emb  # [-1, channel]
        entity_agg_2 = scatter_mean(src=neigh_relation_emb, index=head[edge_idx_same], dim_size=n_entities, dim=0)
        entity_agg_2 = self.activation(self.W2(entity_agg_2)) / 2

        rel_ = scatter_mean(src=relation_weight[edge_type], index=head, dim_size=n_entities, dim=0)
        return torch.add(entity_agg_1, entity_agg_2), rel_

    def forward(self, entity_emb, user_emb, user_emb_cf, item_emb_cf,
                edge_index, edge_type, interact_mat, relation_weight):

        """KG aggregate"""
        entity_agg, rel_ = self.KG_forward(entity_emb, edge_index, edge_type, relation_weight)
        emb_size = entity_emb.shape[1]

        """user aggregate"""
        mat_row = interact_mat[:, 0]
        mat_col = interact_mat[:, 1]

        item_emb_kg = entity_emb[:self.n_items]

        rel_ = rel_[:self.n_items]

        u, ucf = None, None
        for i in range(self.max_iter):
            if u is None:
                p = torch.sigmoid(torch.sum(torch.mul(user_emb[mat_row] * rel_[mat_col], item_emb_kg[mat_col]), dim=1))
                pcf = torch.sigmoid(torch.sum(torch.mul(user_emb_cf[mat_row], item_emb_cf[mat_col]), dim=1))
            else:
                p = torch.sigmoid(torch.sum(torch.mul(u[mat_row] * rel_[mat_col], item_emb_kg[mat_col]), dim=1))
                pcf = torch.sigmoid(torch.sum(torch.mul(ucf[mat_row], item_emb_cf[mat_col]), dim=1))
            p = scatter_softmax(src=p, index=mat_row, dim=0)
            pcf = scatter_softmax(src=pcf, index=mat_row, dim=0)
            mask = (torch.abs(torch.sigmoid(p) - torch.sigmoid(pcf)) < self.gamma).type(torch.int).view(-1, 1)

            u = scatter_sum(src=item_emb_kg[mat_col] * p.view(-1, 1) * mask, index=mat_row, dim_size=self.n_users,
                            dim=0)
            ucf = scatter_sum(src=item_emb_cf[mat_col] * pcf.view(-1, 1) * mask, index=mat_row, dim_size=self.n_users,
                              dim=0)
            if i < self.max_iter - 1:
                u = F.normalize(u, dim=1)
                ucf = F.normalize(ucf, dim=1)

        item_agg = scatter_mean(src=user_emb_cf[mat_row], index=mat_col, dim_size=self.n_items, dim=0)

        return entity_agg, u, ucf, item_agg, mask.detach()


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, channel, n_hops, n_users,
                 n_items, n_relations, interact_mat, gamma, max_iter,
                 device, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.device = device

        relation_weight = nn.init.xavier_uniform_(torch.empty(n_relations, channel))  # not include interact
        self.relation_weight = nn.Parameter(relation_weight)  # [n_relations - 1, in_channel]

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users, n_items=n_items, gamma=gamma, max_iter=max_iter).to(self.device))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _edge_sampling_01(self, edge_index, edge_type, rate=0.5):
        n_edges = edge_index.shape[1]
        m = np.random.choice([0, 1], size=n_edges, p=[0.0, 1.0])
        return m

    def _sparse_dropout(self, x, rate=0.5):
        n_row = x.shape[0]
        random_indices = np.random.choice(n_row, size=int(n_row * rate), replace=False)
        random_indices = torch.LongTensor(random_indices)
        return x[random_indices, :], random_indices

    def forward(self, user_emb, entity_emb, emb_cf, edge_index, edge_type,
                interact_mat, KG_DropEdge_para, mess_dropout=True, node_dropout=False):

        """node dropout"""
        random_idx = None
        m = torch.ones_like(edge_type).to(self.device)
        if node_dropout:
            # edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            m = self._edge_sampling_01(edge_index, edge_type, self.node_dropout_rate)
            m = torch.IntTensor(m).to(self.device)
            interact_mat, random_idx = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        user_emb_cf_res = emb_cf[:self.n_users]
        item_emb_cf_res = emb_cf[self.n_users:]
        user_emb_cf = emb_cf[:self.n_users]
        item_emb_cf = emb_cf[self.n_users:]

        for i in range(len(self.convs)):
            m = m & KG_DropEdge_para[i]
            edge_index1, edge_type1 = edge_index[:, m == 1], edge_type[m == 1]

            entity_emb, user_emb, user_emb_cf, item_emb_cf, bank = self.convs[i](entity_emb, user_emb, user_emb_cf,
                                                                                 item_emb_cf,
                                                                                 edge_index1, edge_type1, interact_mat,
                                                                                 self.relation_weight)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
                user_emb_cf = self.dropout(user_emb_cf)
                item_emb_cf = self.dropout(item_emb_cf)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            user_emb_cf = F.normalize(user_emb_cf)
            item_emb_cf = F.normalize(item_emb_cf)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)
            user_emb_cf_res = torch.add(user_emb_cf_res, user_emb_cf)
            item_emb_cf_res = torch.add(item_emb_cf_res, item_emb_cf)

        return entity_res_emb, user_res_emb, user_emb_cf_res, item_emb_cf_res, [random_idx, bank.view(-1)]


class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, train_cf):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.margin_ccl = args_config.margin
        self.num_neg_sample = args_config.num_neg_sample
        self.gamma = args_config.gamma
        self.max_iter = args_config.max_iter
        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.loss_f = args_config.loss_f
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")
        self.train_cf = train_cf.to(self.device)
        self.similarity_bank = torch.ones([train_cf.shape[0]]).to(self.device)
        self.edge_index, self.edge_type = self._get_edges(graph)

        self._init_weight()
        self._init_loss_function()

        self.gcn = self._init_model()

        # binary drop matrix init
        self.KG_DropEdge_para = np.random.uniform(0.5, 1, size=(self.context_hops, self.edge_type.shape[0]))
        self.KG_DropEdge_para = np.log((self.KG_DropEdge_para + 1e-7) / (1 - self.KG_DropEdge_para + 1e-7))
        self.KG_DropEdge_para = torch.tensor(self.KG_DropEdge_para, requires_grad=False, device='cuda',
                                             dtype=torch.float)

        self.optimizer_stru = torch.optim.Adam([self.KG_DropEdge_para], lr=0.0001)

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.all_embed = nn.Parameter(self.all_embed)
        self.emb_cf = initializer(torch.empty(self.n_items + self.n_users, self.emb_size))
        self.emb_cf = nn.Parameter(self.emb_cf)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_items=self.n_items,
                         n_relations=self.n_relations,
                         interact_mat=self.train_cf,
                         gamma=self.gamma,
                         max_iter=self.max_iter,
                         device=self.device,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def _init_loss_function(self):
        if self.loss_f == "inner_bpr":
            self.loss = self.create_inner_bpr_loss
        elif self.loss_f == 'contrastive_loss':
            self.loss = self.create_contrastive_loss
        else:
            raise NotImplementedError

    def generate_b(self, param):
        u_noise = torch.rand(size=param.shape).to(self.device)
        P1 = torch.sigmoid(-param)
        E1 = (u_noise > P1).type(torch.int).to(self.device)

        P2 = 1 - P1
        E2 = (u_noise < P2).type(torch.int).to(self.device)

        return E1, E2, u_noise

    def bernulli_sample(self, params):
        strus = torch.sigmoid(params)
        strus = torch.bernoulli(strus).to(self.device)
        # print(torch.sum(strus[0]), torch.sum(strus[1]), torch.sum(strus[2]))

        return strus.type(torch.int)

    def gcn_forword(self, idx, user, pos_item, neg_item, user_emb, entity_emb, KG_DropEdge_para):
        entity_gcn_emb, user_gcn_emb, user_gcn_emb_cf, item_gcn_emb_cf, batch_simi_bank = self.gcn(user_emb,
                                                                                                   entity_emb,
                                                                                                   self.emb_cf,
                                                                                                   self.edge_index,
                                                                                                   self.edge_type,
                                                                                                   self.train_cf,
                                                                                                   KG_DropEdge_para=KG_DropEdge_para,
                                                                                                   mess_dropout=self.mess_dropout,
                                                                                                   node_dropout=self.node_dropout)

        self.similarity_bank[batch_simi_bank[0]] = batch_simi_bank[1].type(torch.float32)
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        loss1 = self.loss(u_e, pos_e, neg_e, idx)
        u_e = user_gcn_emb_cf[user]
        pos_e, neg_e = item_gcn_emb_cf[pos_item], item_gcn_emb_cf[neg_item]
        loss2 = self.loss(u_e, pos_e, neg_e, idx)
        return loss1 + loss2

    def forward(self, batch=None):
        idx = batch['pos_index']
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items'].reshape(-1)

        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]

        binary_matrix = self.bernulli_sample(self.KG_DropEdge_para)

        # update drop_matrix para
        first, second, noise = self.generate_b(self.KG_DropEdge_para)
        loss_1 = self.gcn_forword(idx, user, pos_item, neg_item, user_emb, entity_emb, first)
        loss_2 = self.gcn_forword(idx, user, pos_item, neg_item, user_emb, entity_emb, second)

        # disarm_factor = ((1. - first) * second + first * (1. - second)) * (-1.) ** second * torch.sigmoid(torch.abs(self.KG_DropEdge_para))
        a1 = torch.sigmoid(self.KG_DropEdge_para)
        a2 = torch.sigmoid(-self.KG_DropEdge_para)
        disarm_factor = (first - second) * torch.where((a1 - a2) > 0, a1, a2)
        grads = 0.5 * (loss_1 - loss_2) * disarm_factor

        self.optimizer_stru.zero_grad()
        self.KG_DropEdge_para.grad = -grads
        self.optimizer_stru.step()

        # update node and edge embeddings
        loss_network = self.gcn_forword(idx, user, pos_item, neg_item, user_emb, entity_emb, binary_matrix)

        return loss_network

    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]
        binary_matrix = self.bernulli_sample(self.KG_DropEdge_para)
        entity_gcn_emb, user_gcn_emb, user_gcn_emb_cf, item_gcn_emb_cf, _ = self.gcn(user_emb,
                                                                                     entity_emb,
                                                                                     self.emb_cf,
                                                                                     self.edge_index,
                                                                                     self.edge_type,
                                                                                     self.train_cf,
                                                                                     KG_DropEdge_para=binary_matrix,
                                                                                     mess_dropout=False,
                                                                                     node_dropout=False)

        entity_gcn_emb = torch.cat([entity_gcn_emb[:self.n_items], item_gcn_emb_cf], dim=1)
        user_gcn_emb = torch.cat([user_gcn_emb, user_gcn_emb_cf], dim=1)
        return entity_gcn_emb, user_gcn_emb

    def rating(self, u_g_embeddings, i_g_embeddings):
        if self.loss_f == "inner_bpr":
            return torch.matmul(u_g_embeddings, i_g_embeddings.t()).detach().cpu()

        elif self.loss_f == 'contrastive_loss':
            # u_g_embeddings = F.normalize(u_g_embeddings)
            # i_g_embeddings = F.normalize(i_g_embeddings)
            return torch.cosine_similarity(u_g_embeddings[:, :self.emb_size].unsqueeze(1),
                                           i_g_embeddings[:, :self.emb_size].unsqueeze(0), dim=2).detach().cpu() + \
                   torch.cosine_similarity(u_g_embeddings[:, self.emb_size:].unsqueeze(1),
                                           i_g_embeddings[:, self.emb_size:].unsqueeze(0), dim=2).detach().cpu()


    def create_contrastive_loss(self, u_e, pos_e, neg_e, idx):
        batch_size = u_e.shape[0]

        u_e = F.normalize(u_e)
        pos_e = F.normalize(pos_e)
        neg_e = F.normalize(neg_e)

        ui_pos_loss1 = torch.relu(1 - torch.cosine_similarity(u_e, pos_e, dim=1))

        users_batch = torch.repeat_interleave(u_e, self.num_neg_sample, dim=0)

        ui_neg1 = torch.relu(torch.cosine_similarity(users_batch, neg_e, dim=1) - self.margin_ccl)
        ui_neg1 = ui_neg1.view(batch_size, -1)
        x = ui_neg1 > 0
        ui_neg_loss1 = torch.sum(ui_neg1, dim=-1) / (torch.sum(x, dim=-1) + 1e-5)

        loss = ui_pos_loss1 * self.similarity_bank[idx] + ui_neg_loss1

        return loss.mean()


    def create_inner_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        cf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return cf_loss + emb_loss
