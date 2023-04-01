import sys
import os
import argparse
import json

def accuracy(data, target='pred_label'):
    total = 0
    correct = 0
    for d in data:
        if d['label'] == d[target]:
            correct+=1
        total +=1
    return correct/total

def fool_rate(data):
    total = 0
    fooled = 0
    for d in data:
        if d['label'] == d['pred_label']:
            total+=1
            if d['att_pred_label'] != d['label']:
                fooled+=1
    return fooled/total

def fool_rate_any(datas):
    # assume we have the same model being attacked (with different attack methods)
    total = 0
    fooled = 0
    for ds in zip(*datas):
        if ds[0]['label'] == ds[0]['pred_label']:
            total+=1
            for d in ds:
                if d['att_pred_label'] != d['label']:
                    fooled+=1
                    break
    return fooled/total


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--json_path', type=str, nargs='+', required=True, help='saved .json file(s)')
    commandLineParser.add_argument('--fool', action='store_true', help='calulate fooling rate')
    commandLineParser.add_argument('--fool_any', action='store_true', help='calulate fooling rate for multiple attack methods')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # load the json object
    datas = []
    for json_path in args.json_path:
        with open(json_path, 'r') as f:
            datas.append(json.load(f))
    if len(args.json_path) == 1:
        data = datas[0]
    
    if args.fool:
        for d in datas:
            print(f'Accuracy of original predictions\t{accuracy(d)}')
            print(f"Accuracy of attacked predictions\t{accuracy(d, target='att_pred_label')}")
            print(f'Fooling Rate\t{fool_rate(d)}')
            print()
    
    if args.fool_any:
        # check if sample can be attacked by any attack method
        print(f'Fooling Rate Any\t{fool_rate_any(datas)}')
        print()
    
    

    