import os
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

class MyDataset(Dataset):
    def __init__(self, train_cf_pairs, train_user_set, n_items, num_neg_sample):
        self.train_cf_pairs = train_cf_pairs
        self.train_user_set = train_user_set
        self.n_items = n_items
        self.num_neg_sample = num_neg_sample

    def __len__(self):
        return len(self.train_cf_pairs)

    def __getitem__(self, idx):
        ui = self.train_cf_pairs[idx]

        u = int(ui[0])
        each_negs = list()
        neg_item = np.random.randint(low=0, high=self.n_items, size=self.num_neg_sample)
        if len(set(neg_item) & set(self.train_user_set[u])) == 0:
            each_negs += list(neg_item)
        else:
            neg_item = list(set(neg_item) - set(self.train_user_set[u]))
            each_negs += neg_item
            while len(each_negs) < self.num_neg_sample:
                n1 = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if n1 not in self.train_user_set[u]:
                    each_negs += [n1]


        return [ui[0], ui[1], each_negs]