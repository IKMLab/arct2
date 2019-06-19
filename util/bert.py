"""Utilities for working with BERT."""
import os
import copy
import math
import torch
import numpy as np
from torch import nn
from pytorch_pretrained_bert import modeling
import glovar


class SingleInputExample(object):

    def __init__(self, guid, text_a, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = None
        self.label = label

    def __repr__(self):
        info = 'Guid:  %s\n' % self.guid
        info += 'Text:  %s\n' % self.text_a
        info += 'Label: %s' % self.label
        return info


class InputExample(object):
    # note: always two sents - concat claim and premise as sent 1.

    def __init__(self, guid, text_a, text_b, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        info = 'Guid:    %s\n' % self.guid
        info += 'Text A:  %s\n' % self.text_a
        info += 'Text b:  %s\n' % self.text_b
        info += 'Label:   %s' % self.label
        return info


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):

    def get_train_examples(self):
        raise NotImplementedError()

    def get_dev_examples(self):
        raise NotImplementedError()

    def get_test_examples(self):
        raise NotImplementedError

    def get_labels(self):
        raise NotImplementedError()


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label] \
            if example.label is not None else None

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal
    # percent of tokens from each, since if one sequence is very short then each
    # token that's truncated likely contains more information than a longer
    # sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class AttentionTrackingBertModel(modeling.BertModel):

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = modeling.BertEmbeddings(config)
        self.encoder = AttentionTrackingBertEncoder(config)
        self.pooler = modeling.BertPooler(config)
        self.apply(self.init_bert_weights)


class AttentionTrackingBertEncoder(modeling.BertEncoder):

    def __init__(self, config):
        super().__init__(config)
        layer = AttentionTrackingBertLayer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _
             in range(config.num_hidden_layers)])


class AttentionTrackingBertLayer(modeling.BertLayer):

    def __init__(self, config):
        super().__init__(config)
        self.attention = AttentionTrackingBertAttention(config)


class AttentionTrackingBertAttention(modeling.BertAttention):

    def __init__(self, config):
        super().__init__(config)
        self.self = AttentionTrackingBertSelfAttention(config)


class AttentionTrackingBertSelfAttention(modeling.BertSelfAttention):

    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def next_save_path():
        base = os.path.join(glovar.DATA_DIR, 'attn_dists')
        if not os.path.exists(base):
            os.mkdir(base)
        n = len([f for f in os.listdir(base) if '.npy' in f])
        return os.path.join(base, '%s.npy' % (n + 1))

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores \
                           / math.sqrt(self.attention_head_size)
        # Apply mask is (precomputed for all layers in BertModel forward())
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attn = attention_probs.detach().cpu().numpy()
        np.save(self.next_save_path(), attn)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] \
                                  + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
