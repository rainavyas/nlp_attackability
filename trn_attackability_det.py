'''
Train system to detect attackable samples (or robust samples)

Note, model_name: linear-bert-base-uncased means we train a linear classifier on top of bert-base-uncased embedding layer
                  fcn-bert-base-uncased means we train a fully connected classifier on top of bert-base-uncased embedding layer
'''

import torch
import torch.nn as nn
import sys
import os
import argparse
import logging

from src.tools.tools import get_default_device, set_seeds
from src.models.model_selector import select_model
from src.data.attackability_data import select_attack_data
from src.training.cont_trainer import ContTrainer as Trainer
from src.models.model_embedding import model_embed

def base_name_creator(args):
    mname = args.model_name
    mname = '-'.join(mname.split('/'))
    base_name = f'attackable_thresh{args.thresh}_{mname}_{args.data_name}_seed{args.seed}'
    if args.robust:
        base_name = f'robust_thresh{args.thresh}_{mname}_{args.data_name}_seed{args.seed}'
    return base_name

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--out_dir', type=str, required=True, help='Specify dir to save model')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. bert-base-uncased')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. rt')
    commandLineParser.add_argument('--perts', type=str, required=True, nargs='+', help='paths to perturbations')
    commandLineParser.add_argument('--thresh', type=float, default=0.2, help="Specify imperceptibility threshold")
    commandLineParser.add_argument('--bs', type=int, default=8, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=5, help="Specify max epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.00001, help="Specify learning rate")
    commandLineParser.add_argument('--sch', type=int, default=[3], nargs='+', help="Specify scheduler cycle, e.g. 10 100 1000")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")
    commandLineParser.add_argument('--num_classes', type=int, default=2, help="Specify number of classes")
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--robust', action='store_true', help='train to identify robust samples')
    commandLineParser.add_argument('--trained_model_path', type=str, required=True, help='path to trained model for embedding linear classifier')
    args = commandLineParser.parse_args()

    set_seeds(args.seed)
    base_name = base_name_creator(args)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/trn_attackability_det.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Initialise logging
    if not os.path.isdir('LOGs'):
        os.mkdir('LOGs')
    fname = f'LOGs/{base_name}.log'
    logging.basicConfig(filename=fname, filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info('LOG created')

    # Get the device
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the training data (use val set to train attackability detector)
    val_data, train_data = select_attack_data(args, args.perts, thresh=args.thresh, robust=args.robust)

    # Get embeddings
    trained_model_name = '-'.join(args.model_name.split('-')[1:])
    train_dl, num_feats = model_embed(train_data, trained_model_name, args.trained_model_path, device, bs=args.bs, shuffle=True, num_classes=args.num_classes)
    val_dl, _ = model_embed(val_data, trained_model_name, args.trained_model_path, device, bs=args.bs, shuffle=False, num_classes=args.num_classes)

    # Initialise model
    if 'linear' in args.model_name:
        model = select_model('linear', num_labels=2, size=num_feats)
    elif 'fcn' in args.model_name:
        model = select_model('fcn', num_labels=2, size=num_feats)
    else:
        raise ValueError("Invalid model name, must include linear or fcn")
    model.to(device)

    # Define learning objects
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.sch)
    criterion = nn.CrossEntropyLoss().to(device)

    # Train
    out_file = f'{args.out_dir}/{base_name}.th'
    trainer = Trainer(device, model, optimizer, criterion, scheduler)
    trainer.train_process(train_dl, val_dl, out_file, max_epochs=args.epochs)