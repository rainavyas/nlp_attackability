from .model_selector import select_model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def model_embed(data, model_name, model_path, device, bs=8, shuffle=False, num_classes=2):

    model = select_model(model_name=model_name, model_path=model_path, num_labels=num_classes)
    model.eval()
    model.to(device)

    # create dataset
    sentences = [d['text'] for d in data]
    ml = model.tokenizer.model_max_length if model.tokenizer.model_max_length < 5000 else 512
    inputs = model.tokenizer(sentences, padding=True, max_length=ml, truncation=True, return_tensors="pt")
    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    y_attack = torch.LongTensor([d['attackability_label'] for d in data])
    ds = TensorDataset(ids, mask, y_attack)
    dl = DataLoader(ds, batch_size=bs, shuffle=False)

    all_features = []
    labels = []
    with torch.no_grad():
        for (id, m, y) in dl:
            id = id.to(device)
            m = m.to(device)
            features = get_features(model_name, model, id, m)
            all_features.append(features.cpu())
            labels.append(y)
    X = torch.cat(all_features, dim=0)
    y = torch.cat(labels, dim=0)
    num_feats = X.size(-1)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle), num_feats

def get_features(model_name, model, id, m):
    if 'roberta' in model_name:
        outputs = model.model.roberta(id, m)
        return outputs[0][:, 0, :].squeeze(dim=-1)
    if 'bert' in model_name:
        outputs = model.model.bert(id, m)
        return outputs[1]
    if 'xlnet' in model_name:
        outputs = model.model.transformer(id, m)
        # model.model.sequence_summary.summary_type = 'last'
        return outputs[0][:, -1, :].squeeze(dim=-1)
