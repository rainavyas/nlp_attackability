'''
Find correlation between adv attack perturbation sizes between models
Or
Find correlation between probability of correct and perturbation size
'''

import torch
import torch.nn as nn
import sys
import os
import argparse
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.attack.attacker import Attacker

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--perts', type=str, required=True, nargs='+', help='paths to perturbations')
    commandLineParser.add_argument('--compare', action='store_true', help='generate comparison plot between perturbations')
    commandLineParser.add_argument('--binary_sweep', action='store_true', help='binary thresh- attackable or not')
    commandLineParser.add_argument('--plot', type=str, required=True, help='file path to plot')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/correlate.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    ps = [torch.load(p) for p in args.perts]
    names = [n.split('/')[-1].split('_')[1] for n in args.perts]

    if args.compare:

        # Assume only two sets of perturbations passed
        p1 = ps[0].tolist()
        p2 = ps[1].tolist()

        # correlations
        pcc, _ = stats.pearsonr(p1, p2)
        spearman, _ = stats.spearmanr(p1, p2)
        print(f'PCC:\t{pcc}\nSpearman:\t{spearman}')

        # # Scatter plot
        # name1 = names[0]
        # name2 = names[1]
        # data = pd.DataFrame.from_dict({name1:p1, name2:p2})
        # sns.jointplot(x = name1, y = name2, kind = "reg", data = data, scatter_kws={'s': 1})
        # plt.savefig(args.plot, bbox_inches='tight')
        # plt.clf()

    if args.binary_sweep:
        sns.set_style("darkgrid")

        for name, p in zip(names, ps):
            threshs, frac_attackable = Attacker.attack_frac_sweep(p)
            plt.plot(threshs, frac_attackable, label=name, linestyle='dashed')

        threshs, frac_attackable = Attacker.attack_frac_sweep_all(ps)
        plt.plot(threshs, frac_attackable, label='uni')
        plt.ylabel('Fraction attackable')
        plt.xlabel('Imperceptibility Threshold')
        plt.legend()

        plt.savefig(args.plot, bbox_inches='tight')
        plt.clf()
