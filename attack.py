import torch
import torch.nn as nn
import sys
import os
import argparse
import json

from src.tools.tools import get_default_device
from src.models.model_selector import select_model
from src.data.data_selector import select_data
from src.attack.attacker import Attacker
from src.attack.base_attacker import BaseAttacker

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--out_dir', type=str, required=True, help='dir to save output file as .pt file or .json')
    commandLineParser.add_argument('--model_path', type=str, required=True, help='trained model path')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. bert-base-uncased')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. sst')
    commandLineParser.add_argument('--num_classes', type=int, default=2, help="Specify number of classes in data")
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--val', action='store_true', help='apply attack to validation data')
    commandLineParser.add_argument('--attack_method', type=str, default='textfooler', help="Specify attack method")
    commandLineParser.add_argument('--batch', action='store_true', help='apply attack to batch of the data')
    commandLineParser.add_argument('--start', type=int, required=False, default=0, help='start of batch')
    commandLineParser.add_argument('--end', type=int, required=False, default=1000, help='end of batch')
    commandLineParser.add_argument('--min_pert', action='store_true', help='Calculate minimum perturbation size per sample')
    commandLineParser.add_argument('--attack', action='store_true', help='Apply default attack and save attacked samples')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

#     # Get the device
#     if args.force_cpu:
#         device = torch.device('cpu')
#     else:
#         device = get_default_device()

    # Load the test data or validation data
    if not args.val:
        data = select_data(args, train=False)
    else:
        data, _ = select_data(args, train=True)
    if args.batch:
        data = data[args.start:args.end]

    # Load model
    model = select_model(args.model_name, model_path=args.model_path, num_labels=args.num_classes)
#     model.to(device)

    if args.min_pert:

        # Get minimum perturbation sizes per sample
        sentences = [d['text'] for d in data]
        if args.attack_method == 'bae':
            sizes = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4]
        else:
            sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        perts = Attacker.get_all_pert_sizes(sentences, model, method=args.attack_method, sizes=sizes)
        perts = torch.Tensor(perts)

        # Report mean and standard deviation
        print(f'Mean: {torch.mean(perts)}\tStd: {torch.std(perts)}')

        # Save the perturbation sizes
        mname = args.model_name
        mname = '-'.join(mname.split('/'))
        out_file = f'{args.out_dir}/{args.attack_method}_{mname}_{args.data_name}.pt'
        if args.batch:
            out_file = f'{args.out_dir}/{args.attack_method}_{mname}_{args.data_name}_{args.start}-{args.end}.pt'
        torch.save(perts, out_file)
    

    if args.attack:

        # attack each sample with default constraints and save the attack output
    
        # Attack all samples
        att_data = BaseAttacker.attack_all(data, model, method=args.attack_method)

        # Save
        mname = args.model_name
        mname = '-'.join(mname.split('/'))
        out_file = f'{args.out_dir}/{args.attack_method}_{mname}_{args.data_name}.json'
        if args.batch:
            out_file = f'{args.out_dir}/{args.attack_method}_{mname}_{args.data_name}_{args.start}-{args.end}.json'
        with open(out_file, 'w') as f:
            json.dump(att_data, f)
