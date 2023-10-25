import torch
import numpy as np
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from datasets import Mutag, MNIST75sp #, BM

def get_data_loaders(dataset_name, batch_size, batch_size_test=None, random_state=0, data_dir='data'):
    assert dataset_name in ['mutag', 'mnist0', 'mnist1'] # , 'bm_mn', 'bm_ms', 'bm_mt'

    if batch_size_test is None:
        batch_size_test = batch_size

    elif dataset_name == 'mutag':
        dataset = Mutag(root=data_dir + '/mutag')
        dataset.data.y = dataset.data.y.squeeze()
        dataset.data.y = 1 - dataset.data.y # we make the original class "0" as anomalies here
        split_idx = get_random_split_idx(dataset, random_state)
        loaders = get_loaders_mutag(batch_size, batch_size_test, dataset=dataset, split_idx=split_idx)
        num_feat = dataset.data.x.shape[1]
        num_edge_feat = 0

    elif dataset_name in ['mnist0', 'mnist1']:
        num_train, num_test_normal, num_test_anomaly = 1000, 400, 100
        if dataset_name == 'mnist0':
            normal_class = 0
        else:
            normal_class = 1
        train = MNIST75sp(root=data_dir + '/mnist', mode='train')
        test = MNIST75sp(root=data_dir + '/mnist', mode='test')
        loaders = get_loaders_mnist(batch_size, batch_size_test, train, test,
                                    normal_class, num_train, num_test_normal, num_test_anomaly, random_state)
        num_feat = train.data.x.shape[1]
        num_edge_feat = 0

    elif 'bm' in dataset_name:
        pattern = dataset_name[3:]
        transform = T.Compose([T.ToUndirected()])
        train = BM(root=data_dir + '/' + dataset_name, pattern=pattern, mode='train', pre_transform=transform)
        test = BM(root=data_dir + '/' + dataset_name, pattern=pattern, mode='test', pre_transform=transform)
        loaders = get_loaders_bm(batch_size, batch_size_test, train, test)
        num_feat = train.data.x.shape[1]
        num_edge_feat = 8

    meta = {'num_feat':num_feat, 'num_edge_feat':num_edge_feat}

    return loaders, meta


def get_random_split_idx(dataset, random_state=None, test_per=0.1, classification_mode=False):
    if random_state is not None:
        np.random.seed(random_state)

    print('[INFO] Randomly split dataset!')
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    n_test = int(test_per * len(idx))
    test_idx = idx[:n_test]
    train_idx_raw = idx[n_test:]
    normal_mask = (dataset.data.y[train_idx_raw] == 0).numpy()
    if classification_mode:
        train_idx = train_idx_raw #[normal_mask]
    else:
        train_idx = train_idx_raw[normal_mask]

    ano_mask_test = (dataset.data.y[test_idx] == 1).numpy()
    explain_idx = test_idx[ano_mask_test]

    return {'train': train_idx, 'test': test_idx, 'explain': explain_idx}


def get_loaders_mutag(batch_size, batch_size_test, dataset, split_idx=None):
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size_test, shuffle=False)
    explain_loader = DataLoader(dataset[split_idx["explain"]], batch_size=batch_size_test, shuffle=False)

    return {'train': train_loader, 'test': test_loader, 'explain': explain_loader}


def get_loaders_mnist(batch_size, batch_size_test, train_dataset, test_dataset, normal_class,
                      num_train, num_test_normal, num_test_anomaly, random_state, classification_mode=False):
    if random_state is not None:
        np.random.seed(random_state)

    train_dataset.data.y = (train_dataset.data.y != normal_class).type(torch.int64)
    test_dataset.data.y = (test_dataset.data.y != normal_class).type(torch.int64)

    print('[INFO] Randomly split dataset!')
    train_idx = np.arange(len(train_dataset))
    np.random.shuffle(train_idx)
    test_idx = np.arange(len(test_dataset))
    np.random.shuffle(test_idx)

    # test_idx = test_idx[:num_test]
    normal_mask_te = (test_dataset.data.y[test_idx] == 0).numpy()
    test_idx_normal = test_idx[normal_mask_te]
    test_idx_normal = test_idx_normal[:num_test_normal]
    test_idx_ano = test_idx[~normal_mask_te]
    test_idx_ano = test_idx_ano[:num_test_anomaly]
    test_idx = np.concatenate([test_idx_normal,test_idx_ano])

    normal_mask_tr = (train_dataset.data.y[train_idx] == 0).numpy()
    if not classification_mode:
        train_idx = train_idx[normal_mask_tr]
    train_idx = train_idx[:num_train]
    train_loader = DataLoader(train_dataset[train_idx], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset[test_idx], batch_size=batch_size_test, shuffle=False)
    explain_loader = DataLoader(test_dataset[test_idx], batch_size=batch_size_test, shuffle=False)

    return {'train': train_loader, 'test': test_loader, 'explain': explain_loader}


def get_loaders_bm(batch_size, batch_size_test, train_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    explain_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    return {'train': train_loader, 'test': test_loader, 'explain': explain_loader}