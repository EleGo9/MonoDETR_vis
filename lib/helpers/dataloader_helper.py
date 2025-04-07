import torch
import numpy as np
from torch.utils.data import DataLoader
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.datasets.kitti.indy_dataset import INDY_Dataset


# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg, workers=4):
    # perpare dataset
    if cfg['type'] == 'KITTI':
        train_set = KITTI_Dataset(split=cfg['train_split'], cfg=cfg)
        test_set = KITTI_Dataset(split=cfg['test_split'], cfg=cfg)
    elif cfg['type'] == 'INDY':
        train_sets = []
        test_sets = []
        root_dirs_train = cfg['root_dir_train']
        for root_dir in root_dirs:
            train_sets.append(INDY_Dataset(split=cfg['train_split'], cfg=cfg, root_dir=root_dir))
            print('training: ', root_dir)
        root_dirs_test = cfg['root_dir_test']
        for root_dir_test in root_dirs_test:
            test_sets.append(INDY_Dataset(split=cfg['test_split'], cfg=cfg, root_dir = root_dir_test))
            print('test: ', root_dir_test)

        train_set = torch.utils.data.ConcatDataset(train_sets)
        test_set = torch.utils.data.ConcatDataset(test_sets)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    # prepare dataloader
    train_loader = DataLoader(dataset=train_set,
                              batch_size=cfg['batch_size'],
                              num_workers=workers,
                              worker_init_fn=my_worker_init_fn,
                              shuffle=True,
                              pin_memory=False,
                              drop_last=False)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=cfg['batch_size'],
                             num_workers=workers,
                             worker_init_fn=my_worker_init_fn,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    return train_loader, test_loader
