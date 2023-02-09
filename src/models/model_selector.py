from .models import SequenceClassifier
from .head import SingleLinear, FCN

import torch

def select_model(model_name='bert-base-uncased', model_path=None, pretrained=True, num_labels=2, size=768):
    if model_name == 'linear':
        model = SingleLinear(size, num_labels)
    elif model_name == 'fcn':
        model = FCN(size, num_labels)
    else:
        model =  SequenceClassifier(model_name=model_name, pretrained=pretrained, num_labels=num_labels)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    return model