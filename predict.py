'''
Save the predictions from a model in a file
- save a torch tensor: num_samples x num_classes
'''

import torch
import torch.nn as nn
import sys
import os
import argparse

from src.tools.tools import get_default_device
from src.models.model_selector import select_model
from src.data.data_selector import select_data
from src.training.batch_trainer import BatchTrainer as Trainer

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--out_dir', type=str, required=True, help='Specify dir to save predictions')
    commandLineParser.add_argument('--model_path', type=str, required=True, help='e.g. experiments/trained_models/my_model.th')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. bert-base-uncased')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. rt')
    commandLineParser.add_argument('--bs', type=int, default=8, help="Specify batch size")
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--num_classes', type=int, default=2, help="Specify number of classes")
    commandLineParser.add_argument('--use_val', action='store_true', help='use validation data or test data for predictions')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/predict.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load model
    model = select_model(args.model_name, args.model_path, num_labels=args.num_classes)
    model.to(device)

    # Load data
    if args.use_val:
        data, _ = select_data(args, train=True)
    else:
        data = select_data(args, train=False)
    dl = Trainer.prep_dl(model, data, bs=args.bs, shuffle=False)

        # Get probability predictions
    criterion = nn.CrossEntropyLoss().to(device)
    logits = Trainer.eval(dl, model, criterion, device, return_logits=True)
    s = torch.nn.Softmax(dim=1)
    probs = s(logits)

    # Save
    mname = args.model_name
    mname = '-'.join(mname.split('/'))
    out_file = f'{args.out_dir}/model_{mname}_data_{args.data_name}.pt'
    torch.save(probs, out_file)