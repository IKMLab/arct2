"""Utility code for working with SST-2."""
import os
import pandas as pd
import torch
from torch import nn
import pytorch_pretrained_bert as ppb
from pytorch_pretrained_bert import modeling
import glovar


def load(dataset):
    path = os.path.join(glovar.SST_DIR, '%s.tsv' % dataset)
    return pd.read_csv(path, sep='\t')


# processor, loader, collate, etc...


class BERT(modeling.PreTrainedBertModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.bert = ppb.BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits
