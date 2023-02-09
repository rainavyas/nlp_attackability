'''
Evaluate performance of the trained attackability/robust detector
'''

'''
generate pr curve over test data using trained attackability detector (on val)
-> can generate pr curve for unattackability if desired
'''

import torch
import torch.nn as nn
import sys
import os
import argparse
from sklearn.metrics import precision_recall_curve
import numpy as np

from src.tools.tools import get_default_device, get_best_f_score
from src.models.model_selector import select_model
from src.data.attackability_data import select_attack_data
from src.training.cont_trainer import ContTrainer as Trainer
from src.models.model_embedding import model_embed

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--model_paths', type=str, nargs='+', required=True, help='Specify trained attackability models, list if ensemble')
    commandLineParser.add_argument('--model_names', type=str, nargs='+', required=True, help='e.g. vgg16, list multiple if ensemble of detectors')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. sst')
    commandLineParser.add_argument('--perts', type=str, required=True, nargs='+', help='paths to perturbations')
    commandLineParser.add_argument('--thresh', type=float, default=0.2, help="Specify imperceptibility threshold")
    commandLineParser.add_argument('--bs', type=int, default=64, help="Specify batch size")
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--robust', action='store_true', help='train to identify robust samples')
    commandLineParser.add_argument('--trained_model_paths', type=str, nargs='+', required=True, help='paths to trained models for embedding linear classifiers')
    commandLineParser.add_argument('--num_classes', type=int, default=10, help="Specify number of classes in data for trained_model_paths")
    commandLineParser.add_argument('--spec', action='store_true', help='if mulitple models passed in perts, last model is target. Label attackable sample only if attackable for target, but not universally.')
    commandLineParser.add_argument('--vspec', action='store_true', help='if mulitple models passed in perts, last model is target. Label attackable sample only if attackable for target ONLY - no other model.')
    commandLineParser.add_argument('--pr_save_path', type=str, default='', help='path to save raw pr values for later plotting')
    commandLineParser.add_argument('--combination', type=str, default='sum', choices=['sum', 'product'], help="method to combine ensemble of detector probabilities")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/evl_attackability_det.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the attacked test data
    data = select_attack_data(args, args.perts, thresh=args.thresh, use_val=False, val_for_train=False, spec=args.spec, vspec=args.vspec, robust=args.robust)

    dls = []
    num_featss = []
    for mname, mpath in zip(args.model_names, args.trained_model_paths):
        trained_model_name = '-'.join(mname('-')[1:])
        # Get embeddings per model
        trained_model_name = mname.split('-')[-1]
        dl, num_feats = model_embed(data, trained_model_name, args.trained_model_path, device, bs=args.bs, shuffle=False, num_classes=args.num_classes)
        dls.append(dl)
        num_featss.append(num_feats)


    # Load models
    models = []
    for mname, mpath, n in zip(args.model_names, args.model_paths, num_featss):
        if 'linear' in mname:
            model = select_model('linear', model_path=mpath, num_classes=2, size=n)
        elif 'fcn' in mname:
            model = select_model('fcn', model_path=mpath, num_classes=2, size=n)
        else:
            raise ValueError("Need to be fcn or linear")
        model.to(device)
        models.append(model)

    # Get ensemble probability predictions
    criterion = nn.CrossEntropyLoss().to(device)
    s = torch.nn.Softmax(dim=1)
    all_probs = []
    for dl, model in zip(dls, models):
        logits, labels = Trainer.eval(dl, model, criterion, device, return_logits=True)   
        probs = s(logits)
        all_probs.append(probs)
        labels = labels.detach().cpu().tolist()
        
    if args.combination == 'sum':
        probs = torch.mean(torch.stack(all_probs), dim=0)[:,1].squeeze(dim=-1).detach().cpu().tolist()
    elif args.combination == 'product':
        probs = torch.prod(torch.stack(all_probs), dim=0)[:,1].squeeze(dim=-1).detach().cpu().tolist()

    # Get precision-recall curves
    precision, recall, _ = precision_recall_curve(labels, probs)
    precision = precision[:-1]
    recall = recall[:-1]
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)
    print('Best F1', best_f1)

    if args.pr_save_path != '':
        np.savez(args.pr_save_path, precision=np.asarray(precision), recall=np.asarray(recall))
