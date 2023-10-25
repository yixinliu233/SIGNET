import os
import os.path as osp
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Constant
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import StratifiedKFold


def get_ad_split_TU(args, fold=5):
    DS = args.dataset
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    dataset = TUDataset(path, name=DS)
    data_list = []
    label_list = []

    for data in dataset:
        data_list.append(data)
        label_list.append(data.y.item())

    kfd = StratifiedKFold(n_splits=fold, random_state=0, shuffle=True)

    splits = []
    for k, (train_index, test_index) in enumerate(kfd.split(data_list, label_list)):
        splits.append((train_index, test_index))

    return splits


def get_data_loaders_TU(args, split):
    DS = args.dataset

    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)

    if DS in ['IMDB-BINARY', 'REDDIT-BINARY', 'COLLAB']:
        dataset = TUDataset(path, name=DS, transform=(Constant(1, cat=False)))
    else:
        dataset = TUDataset(path, name=DS)

    dataset_num_features = dataset.num_node_features

    data_list = []
    label_list = []

    for data in dataset:
        data.edge_attr = None
        data_list.append(data)
        label_list.append(data.y.item())

    (train_index, test_index) = split
    data_train_ = [data_list[i] for i in train_index]
    data_test = [data_list[i] for i in test_index]

    data_train = []
    for data in data_train_:
        if data.y != 0:
            data_train.append(data)

    idx = 0
    for data in data_train:
        data.y = 0
        data['idx'] = idx
        idx += 1

    for data in data_test:
        data.y = 1 if data.y == 0 else 0

    dataloader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size_test, shuffle=True)
    meta = {'num_feat':dataset_num_features, 'num_train':len(data_train), 'num_edge_feat':0}
    loader_dict = {'train': dataloader, 'test': dataloader_test}

    return loader_dict, meta
