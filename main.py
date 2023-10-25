import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from models import GIN, Explainer_GIN, HyperGNN, Explainer_MLP
from arguments import arg_parse
from get_data_loaders import get_data_loaders
from get_data_loaders_tuad import get_ad_split_TU, get_data_loaders_TU
import random

import warnings
warnings.filterwarnings("ignore")

explainable_datasets = ['mutag', 'mnist0', 'mnist1', 'bm_mn', 'bm_ms', 'bm_mt']


class SIGNET(nn.Module):
    def __init__(self, input_dim, input_dim_edge, args, device):
        super(SIGNET, self).__init__()

        self.device = device

        self.embedding_dim = args.hidden_dim
        if args.readout == 'concat':
            self.embedding_dim *= args.encoder_layers

        if args.explainer_model == 'mlp':
            self.explainer = Explainer_MLP(input_dim, args.explainer_hidden_dim, args.explainer_layers)
        else:
            self.explainer = Explainer_GIN(input_dim, args.explainer_hidden_dim,
                                           args.explainer_layers, args.explainer_readout)

        self.encoder = GIN(input_dim, args.hidden_dim, args.encoder_layers, args.pooling, args.readout)
        self.encoder_hyper = HyperGNN(input_dim, input_dim_edge, args.hidden_dim, args.encoder_layers, args.pooling, args.readout)

        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_hyper = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        node_imp = self.explainer(data.x, data.edge_index, data.batch)
        edge_imp = self.lift_node_score_to_edge_score(node_imp, data.edge_index)

        y, _ = self.encoder(data.x, data.edge_index, data.batch, node_imp)
        y_hyper, _ = self.encoder_hyper(data.x, data.edge_index, data.edge_attr, data.batch, edge_imp)

        y = self.proj_head(y)
        y_hyper = self.proj_head_hyper(y_hyper)

        return y, y_hyper, node_imp, edge_imp

    @staticmethod
    def loss_nce(x1, x2, temperature=0.2):
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-10)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-10)

        loss_0 = - torch.log(loss_0 + 1e-10)
        loss_1 = - torch.log(loss_1 + 1e-10)
        loss = (loss_0 + loss_1) / 2.0
        return loss

    def lift_node_score_to_edge_score(self, node_score, edge_index):
        src_lifted_att = node_score[edge_index[0]]
        dst_lifted_att = node_score[edge_index[1]]
        edge_score = src_lifted_att * dst_lifted_att
        return edge_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run(args, seed, split=None):

    set_seed(seed)
    is_xgad = args.dataset in explainable_datasets

    if is_xgad:
        loaders, meta = get_data_loaders(args.dataset, args.batch_size, args.batch_size_test, random_state=seed)
    else:
        loaders, meta = get_data_loaders_TU(args, split)
    n_feat = meta['num_feat']
    n_edge_feat = meta['num_edge_feat']

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    model = SIGNET(n_feat, n_edge_feat, args, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = loaders['train']
    test_loader = loaders['test']

    if is_xgad:
        explain_loader = loaders['explain']

    for epoch in range(1, args.epochs+1):

        model.train()
        loss_all = 0
        num_sample = 0
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            y, y_hyper, node_imp, edge_imp = model(data)
            loss = model.loss_nce(y, y_hyper).mean()
            loss_all += loss.item() * data.num_graphs
            num_sample += data.num_graphs
            loss.backward()
            optimizer.step()

        info_train = 'Epoch {:3d}, Loss CL {:.4f}'.format(epoch, loss_all / num_sample)

        if epoch % args.log_interval == 0:
            model.eval()

            # anomaly detection
            all_ad_true = []
            all_ad_score = []
            for data in test_loader:
                all_ad_true.append(data.y.cpu())
                data = data.to(device)
                with torch.no_grad():
                    y, y_hyper, _, _ = model(data)
                    ano_score = model.loss_nce(y, y_hyper)
                all_ad_score.append(ano_score.cpu())

            ad_true = torch.cat(all_ad_true)
            ad_score = torch.cat(all_ad_score)
            ad_auc = roc_auc_score(ad_true, ad_score)

            info_test = 'AD_AUC:{:.4f}'.format(ad_auc)

            # explanation
            if is_xgad:
                all_node_explain_true = []
                all_node_explain_score = []
                all_edge_explain_true = []
                all_edge_explain_score = []
                for data in explain_loader:
                    data = data.to(device)
                    with torch.no_grad():
                        node_score = model.explainer(data.x, data.edge_index, data.batch)
                        edge_score = model.lift_node_score_to_edge_score(node_score, data.edge_index)
                    all_node_explain_true.append(data.node_label.cpu())
                    all_node_explain_score.append(node_score.cpu())
                    all_edge_explain_true.append(data.edge_label.cpu())
                    all_edge_explain_score.append(edge_score.cpu())

                x_node_true = torch.cat(all_node_explain_true)
                x_node_score = torch.cat(all_node_explain_score)
                x_node_auc = roc_auc_score(x_node_true, x_node_score)

                x_edge_true = torch.cat(all_edge_explain_true)
                x_edge_score = torch.cat(all_edge_explain_score)
                x_edge_auc = roc_auc_score(x_edge_true, x_edge_score)

                info_test += '| X AUC(node):{:.4f} | X AUC(edge):{:.4f}'.format(x_node_auc, x_edge_auc)

            print(info_train + '   ' + info_test)

    if is_xgad:
        return ad_auc, x_node_auc, x_edge_auc
    else:
        return ad_auc

if __name__ == '__main__':
    args = arg_parse()
    ad_aucs = []
    if args.dataset in explainable_datasets:
        x_node_aucs = []
        x_edge_aucs = []
        splits = [None] * args.num_trials
    else:
        splits = get_ad_split_TU(args, fold=5)
        key_auc_list = []

    for trial in range(args.num_trials):
        results = run(args, seed=trial, split=splits[trial])
        if args.dataset in explainable_datasets:
            ad_auc, x_node_auc, x_edge_auc = results
            ad_aucs.append(ad_auc)
            x_node_aucs.append(x_node_auc)
            x_edge_aucs.append(x_edge_auc)
        else:
            ad_auc = results
            ad_aucs.append(ad_auc)

    results = 'AUC: {:.2f}+-{:.2f}'.format(np.mean(ad_aucs) * 100, np.std(ad_aucs) * 100)
    if args.dataset in explainable_datasets:
        results += ' | X AUC (node): {:.2f}+-{:.2f}'.format(np.mean(x_node_aucs) * 100, np.std(x_node_aucs) * 100)
        results += ' | X AUC (edge): {:.2f}+-{:.2f}'.format(np.mean(x_edge_aucs) * 100, np.std(x_edge_aucs) * 100)

    print('[FINAL RESULTS] ' + results)