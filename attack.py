import torch
import torch.nn as nn
import sys
import os
import argparse

from src.tools.tools import get_default_device
from src.models.model_selector import select_model
from src.data.data_selector import select_data
from src.attack.attacker import Attacker

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--out_dir', type=str, required=True, help='dir to save output file as .pt file')
    commandLineParser.add_argument('--model_path', type=str, required=True, help='trained model path')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. bert-base-uncased')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. sst')
    commandLineParser.add_argument('--num_classes', type=int, default=2, help="Specify number of classes in data")
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--val', action='store_true', help='apply attack to validation data')
    commandLineParser.add_argument('--attack_method', type=str, default='textfooler', help="Specify attack method")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the test data or validation data
    if not args.val:
        data = select_data(args, train=False)
    else:
        data, _ = select_data(args, train=True)

    # Load model
    model = select_model(args.model_name, model_path=args.model_path, num_labels=args.num_classes)
    model.to(device)

    # Get minimum perturbation sizes per sample
    perts = Attacker.get_all_pert_sizes([d['text'] for d in data], model, method=args.attack_method)
    perts = torch.Tensor(perts)

    # Report mean and standard deviation
    print(f'Mean: {torch.mean(perts)}\tStd: {torch.std(perts)}')

    # Save the perturbation sizes
    mname = args.model_name
    mname = '-'.join(mname.split('/'))
    out_file = f'{args.out_dir}/{args.attack_method}_{mname}_{args.data_name}.pt'
    torch.save(perts, out_file)
