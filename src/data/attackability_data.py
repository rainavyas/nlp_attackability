import torch

from .data_selector import select_data
from sklearn.model_selection import train_test_split

def select_attack_data(args, pert_paths, thresh=0.2, val=0.2, use_val=True, val_for_train=True, spec=False, vspec=False, robust=False):
    '''
    use_val -> use validation data
    val_for_train -> split the selected validation data into further trn and val splits

    For a single sample:
        if ALL model perturbations are smaller than threshold => attackable -> label 1
        Otherwise -> label 0.
        If spec: 
        if mulitple models passed in pert_paths, last model is target. Return attackable sample only if attackable for target, but not universally attackable for all models.
        If vspec:
        if mulitple models passed in perts, last model is target. Label attackable sample only if attackable for target ONLY - no other models.
        If robust is True, then same thing as attackability but flipped.
    '''
    ps = [torch.load(p) for p in pert_paths]

    if robust:
        attackability_labels = robust_labels(ps, thresh, spec=spec, vspec=vspec)
    else:
        attackability_labels = attackable_labels(ps, thresh, spec=spec, vspec=vspec)
    
    if use_val:
        data, _ = select_data(args, train=True)
    else:
        data = select_data(args, train=False)

    for d, a in zip(data, attackability_labels):
        d['attackability_label'] = a

    if val_for_train:
        # split into train and validation
        num_val = int(val*len(data))
        train_indices, val_indices = train_test_split(range(len(data)), test_size=num_val, random_state=42)
        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]

        return val_data, train_data
    else:
        return data

def attackable_labels(ps, thresh, spec=False, vspec=False):
    '''attackable samples'''
    attackability_labels = []

    for sample in zip(*ps):
        num_attackable = 0
        for pert in sample:
            if pert <= thresh:
                num_attackable += 1

        if spec:
            if num_attackable == len(sample) or sample[-1] > thresh:
                attackability_labels.append(0)
            else:
                attackability_labels.append(1)
        elif vspec:
            if num_attackable == 1 and sample[-1] <= thresh:
                attackability_labels.append(1)
            else:
                attackability_labels.append(0)
        else:
            # find universally attackable samples
            if num_attackable == len(sample):
                attackability_labels.append(1)
            else:
                attackability_labels.append(0)

    return attackability_labels

def robust_labels(ps, thresh, spec=False, vspec=False):
    '''robust samples'''
    robust_labels = []

    for sample in zip(*ps):
        num_unattackable = 0
        for pert in sample:
            if pert >= thresh:
                num_unattackable += 1

        if spec:
            if num_unattackable == len(sample) or sample[-1] < thresh:
                robust_labels.append(0)
            else:
                robust_labels.append(1)
        elif vspec:
            if num_unattackable == 1 and sample[-1] >= thresh:
                robust_labels.append(1)
            else:
                robust_labels.append(0)
        else:
            # find universally robust samples
            if num_unattackable == len(sample):
                robust_labels.append(1)
            else:
                robust_labels.append(0)
    return robust_labels