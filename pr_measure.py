'''
Can pre-defined measures be used to identify attackable samples
generate pr curve
'''
import torch
import torch.nn as nn
import sys
import os
import argparse
from sklearn.metrics import precision_recall_curve
import numpy as np

from src.data.attackability_data import select_attack_data
from src.tools.tools import get_best_f_score

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--measure', type=str, required=True, help='e.g. confidence')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. sst')
    commandLineParser.add_argument('--perts', type=str, required=True, nargs='+', help='paths to perturbations')
    commandLineParser.add_argument('--preds', type=str, required=True, nargs='+', help='path to saved model predictions')
    commandLineParser.add_argument('--thresh', type=float, default=0.2, help="Specify imperceptibility threshold")
    commandLineParser.add_argument('--spec', action='store_true', help='if mulitple models passed in perts, last model is target. Label attackable sample only if attackable for target, but not universally.')
    commandLineParser.add_argument('--vspec', action='store_true', help='if mulitple models passed in perts, last model is target. Label attackable sample only if attackable for target ONLY - no other model.')
    commandLineParser.add_argument('--robust', action='store_true', help='train to identify robust samples')
    commandLineParser.add_argument('--pr_save_path', type=str, default='', help='path to save raw pr values for later plotting')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/pr_measure.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the attacked test data labels
    data = select_attack_data(args, args.perts, thresh=args.thresh, use_val=False, val_for_train=False, spec=args.spec, vspec=args.vspec, robust=args.robust)
    labels = []
    for d in data:
        l = d['attackability_label']
        labels.append(l)
    
    # load the value as per measure

    if args.measure == 'confidence':
        # use negative confidence (high negative confidence correlates with attackability)
        preds = [torch.load(p) for p in args.preds]
        measure = []
        for ps in zip(*preds):
            probs = torch.mean(torch.stack(ps), dim=0)
            conf = probs[torch.argmax(probs)].item()
            if not args.robust:
                measure.append(-1*conf)
            else:
                measure.append(conf)

    # plot pr curve
    precision, recall, _ = precision_recall_curve(labels, measure)
    precision = precision[:-1]
    recall = recall[:-1]
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)
    print('Best F1', best_f1)

    if args.pr_save_path != '':
        np.savez(args.pr_save_path, precision=np.asarray(precision), recall=np.asarray(recall))

