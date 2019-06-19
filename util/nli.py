"""Utility code for working with NLI data."""
import os
import pandas as pd
import torch
from torch import nn
import pytorch_pretrained_bert as ppb
from pytorch_pretrained_bert import modeling
import glovar


def load(dataset, subset):
    path = os.path.join(glovar.NLI_DIR, dataset, '%s.tsv' % subset)
    return pd.read_csv(path, sep='\t')


# processer, loader, etc...


class BERT(modeling.PreTrainedBertModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 3
        self.bert = ppb.BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.apply(self.init_bert_weights)
        self.softmax = nn.Softmax(dim=-1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs = self.softmax(logits)
        loss = self.loss(probs.view(-1, self.num_labels), labels.view(-1))
        return loss, logits
