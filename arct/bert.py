"""Utilities for BERT for ARCT."""
import torch
from torch import nn
from torch.utils import data as td
from util import bert, training
from arct import data
import pytorch_pretrained_bert as ppb
from pytorch_pretrained_bert import modeling


#
# Data Preparation


class ARCTDataset(td.Dataset):

    def __init__(self, ids, tensors):
        self.ids = ids
        self.tensors = tensors

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.tensors[i]


class Batch(training.Batch):

    def __init__(self, ids, input_ids_0, input_mask_0, segment_ids_0,
                 input_ids_1, input_mask_1, segment_ids_1, label_ids):
        self.ids = ids
        self.input_ids_0 = torch.cat([x.unsqueeze(0) for x in input_ids_0])
        self.input_mask_0 = torch.cat([x.unsqueeze(0) for x in input_mask_0])
        self.segment_ids_0 = torch.cat([x.unsqueeze(0) for x in segment_ids_0])
        self.input_ids_1 = torch.cat([x.unsqueeze(0) for x in input_ids_1])
        self.input_mask_1 = torch.cat([x.unsqueeze(0) for x in input_mask_1])
        self.segment_ids_1 = torch.cat([x.unsqueeze(0) for x in segment_ids_1])
        self.label_ids = torch.cat([l.unsqueeze(0) for l in label_ids])

    def __len__(self):
        return len(self.ids)

    def tensors(self):
        return self.input_ids_0, self.input_mask_0, self.segment_ids_0, \
               self.input_ids_1, self.input_mask_1, self.segment_ids_1, \
               self.label_ids

    def to(self, device):
        self.input_ids_0 = self.input_ids_0.to(device)
        self.input_mask_0 = self.input_mask_0.to(device)
        self.segment_ids_0 = self.segment_ids_0.to(device)
        self.input_ids_1 = self.input_ids_1.to(device)
        self.input_mask_1 = self.input_mask_1.to(device)
        self.segment_ids_1 = self.segment_ids_1.to(device)
        self.label_ids = self.label_ids.to(device)


def collate(items):
    tensors = [x[1] for x in items]
    return Batch(
        ids=[x[0] for x in items],
        input_ids_0=[x[0] for x in tensors],
        input_mask_0=[x[1] for x in tensors],
        segment_ids_0=[x[2] for x in tensors],
        input_ids_1=[x[3] for x in tensors],
        input_mask_1=[x[4] for x in tensors],
        segment_ids_1=[x[5] for x in tensors],
        label_ids=[x[6] for x in tensors])


class DataLoaders(training.DataLoaders):

    def __init__(self, args):
        self.tokenizer = ppb.BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=True)
        self.n_training_points = None

    def train(self, args):
        df = data.load('train-original')
        self.n_training_points = len(df)
        examples = self.create_examples(df)
        return self.get_data_loader(examples, args)

    def dev(self, args):
        df = data.load('dev-original')
        self.n_train_examples = len(df)
        examples = self.create_examples(df)
        return self.get_data_loader(examples, args)

    def test(self, args):
        df = data.load('test-original')
        self.n_train_examples = len(df)
        examples = self.create_examples(df)
        return self.get_data_loader(examples, args)

    @staticmethod
    def append_claim_reason(claim, reason):
        if claim[-1] != '.':
            claim += '.'
        if reason[-1] != '.':
            reason += '.'
        return '%s %s' % (claim, reason)

    def create_examples(self, df):
        examples = []
        for _, line in df.iterrows():
            guid = line['#id']
            text_a = self.append_claim_reason(line['claim'], line['reason'])
            text_b_0 = line['warrant0']
            text_b_1 = line['warrant1']
            # 0&1 both in fwd together - same label in both InputExamples OK
            label = int(line['correctLabelW0orW1'])
            examples.append([
                bert.InputExample(guid=guid,
                                  text_a=text_a,
                                  text_b=text_b_0,
                                  label=label),
                bert.InputExample(guid=guid,
                                  text_a=text_a,
                                  text_b=text_b_1,
                                  label=label)])
        return examples

    def get_data_loader(self, examples, args):
        features_0 = bert.convert_examples_to_features(
            [x[0] for x in examples], self.get_labels(),
            args.max_seq_length, self.tokenizer)
        features_1 = bert.convert_examples_to_features(
            [x[1] for x in examples], self.get_labels(),
            args.max_seq_length, self.tokenizer)
        features = list(zip(features_0, features_1))
        input_ids_0 = torch.tensor([f[0].input_ids for f in features],
                                   dtype=torch.long)
        input_mask_0 = torch.tensor([f[0].input_mask for f in features],
                                    dtype=torch.long)
        segment_ids_0 = torch.tensor([f[0].segment_ids for f in features],
                                     dtype=torch.long)
        input_ids_1 = torch.tensor([f[1].input_ids for f in features],
                                   dtype=torch.long)
        input_mask_1 = torch.tensor([f[1].input_mask for f in features],
                                    dtype=torch.long)
        segment_ids_1 = torch.tensor([f[1].segment_ids for f in features],
                                     dtype=torch.long)
        label_ids = torch.tensor([f[0].label_id for f in features],
                                 dtype=torch.long)
        ids = [x[0].guid for x in examples]
        tensors = td.TensorDataset(
            input_ids_0, input_mask_0, segment_ids_0,
            input_ids_1, input_mask_1, segment_ids_1,
            label_ids)
        train_data = ARCTDataset(ids, tensors)
        if args.local_rank == -1:
            train_sampler = td.RandomSampler(train_data)
        else:
            train_sampler = td.DistributedSampler(train_data)
        data_loader = td.DataLoader(
            dataset=train_data,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            collate_fn=collate)
        return data_loader

    @staticmethod
    def get_labels():
        return [0, 1]


class DataLoadersAdvOriginal(DataLoaders):
    """Note: due to accident, the original experiments in the paper were
    performed on the `swapped` train and `negated` dev and test sets.

    It turns out the results are the same if we use all swapped.

    But the negated adversarial dataset introduces new spurious statistical
    cues over the claims and warrants - see {model_name}_adv_neg_cw results
    which demonstrate this fact."""

    def train(self, args):
        df = data.load('train-adv-swapped')
        self.n_training_points = len(df)
        examples = self.create_examples(df)
        return self.get_data_loader(examples, args)

    def dev(self, args):
        df = data.load('dev-adv-negated')
        self.n_training_points = len(df)
        examples = self.create_examples(df)
        return self.get_data_loader(examples, args)

    def test(self, args):
        df = data.load('test-negated')
        self.n_training_points = len(df)
        examples = self.create_examples(df)
        return self.get_data_loader(examples, args)


class DataLoadersAdvSwapped(DataLoaders):

    def train(self, args):
        df = data.load('train-adv-swapped')
        self.n_training_points = len(df)
        examples = self.create_examples(df)
        return self.get_data_loader(examples, args)

    def dev(self, args):
        df = data.load('dev-adv-swapped')
        self.n_training_points = len(df)
        examples = self.create_examples(df)
        return self.get_data_loader(examples, args)

    def test(self, args):
        df = data.load('test-adv-swapped')
        self.n_training_points = len(df)
        examples = self.create_examples(df)
        return self.get_data_loader(examples, args)


class DataLoadersAdvSwappedCW(DataLoadersAdvSwapped):

    def create_examples(self, df):
        examples = []
        for _, line in df.iterrows():
            guid = line['#id']
            text_a = line['claim']
            text_b_0 = line['warrant0']
            text_b_1 = line['warrant1']
            label = int(line['correctLabelW0orW1'])
            examples.append([
                bert.InputExample(guid=guid,
                                  text_a=text_a,
                                  text_b=text_b_0,
                                  label=label),
                bert.InputExample(guid=guid,
                                  text_a=text_a,
                                  text_b=text_b_1,
                                  label=label)])
        return examples


class DataLoadersAdvSwappedRW(DataLoadersAdvSwapped):

    def create_examples(self, df):
        examples = []
        for _, line in df.iterrows():
            guid = line['#id']
            text_a = line['reason']
            text_b_0 = line['warrant0']
            text_b_1 = line['warrant1']
            label = int(line['correctLabelW0orW1'])
            examples.append([
                bert.InputExample(guid=guid,
                                  text_a=text_a,
                                  text_b=text_b_0,
                                  label=label),
                bert.InputExample(guid=guid,
                                  text_a=text_a,
                                  text_b=text_b_1,
                                  label=label)])
        return examples


class DataLoadersAdvSwappedW(DataLoadersAdvSwapped):

    def create_examples(self, df):
        examples = []
        for _, line in df.iterrows():
            guid = line['#id']
            text_0 = line['warrant0']
            text_1 = line['warrant1']
            label = int(line['correctLabelW0orW1'])
            examples.append([
                bert.SingleInputExample(guid=guid,
                                        text_a=text_0,
                                        label=label),
                bert.SingleInputExample(guid=guid,
                                        text_a=text_1,
                                        label=label)])
        return examples


class DataLoadersAdvNegated(DataLoaders):

    def train(self, args):
        df = data.load('train-adv-negated')
        self.n_training_points = len(df)
        examples = self.create_examples(df)
        return self.get_data_loader(examples, args)

    def dev(self, args):
        df = data.load('dev-adv-negated')
        self.n_training_points = len(df)
        examples = self.create_examples(df)
        return self.get_data_loader(examples, args)

    def test(self, args):
        df = data.load('test-adv-negated')
        self.n_training_points = len(df)
        examples = self.create_examples(df)
        return self.get_data_loader(examples, args)


class DataLoadersAdvNegatedCW(DataLoadersAdvNegated):

    def create_examples(self, df):
        examples = []
        for _, line in df.iterrows():
            guid = line['#id']
            text_a = line['claim']
            text_b_0 = line['warrant0']
            text_b_1 = line['warrant1']
            label = int(line['correctLabelW0orW1'])
            examples.append([
                bert.InputExample(guid=guid,
                                  text_a=text_a,
                                  text_b=text_b_0,
                                  label=label),
                bert.InputExample(guid=guid,
                                  text_a=text_a,
                                  text_b=text_b_1,
                                  label=label)])
        return examples


class DataLoadersAdvNegatedRW(DataLoadersAdvNegated):

    def create_examples(self, df):
        examples = []
        for _, line in df.iterrows():
            guid = line['#id']
            text_a = line['reason']
            text_b_0 = line['warrant0']
            text_b_1 = line['warrant1']
            label = int(line['correctLabelW0orW1'])
            examples.append([
                bert.InputExample(guid=guid,
                                  text_a=text_a,
                                  text_b=text_b_0,
                                  label=label),
                bert.InputExample(guid=guid,
                                  text_a=text_a,
                                  text_b=text_b_1,
                                  label=label)])
        return examples


class DataLoadersAdvNegatedW(DataLoadersAdvNegated):

    def create_examples(self, df):
        examples = []
        for _, line in df.iterrows():
            guid = line['#id']
            text_0 = line['warrant0']
            text_1 = line['warrant1']
            label = int(line['correctLabelW0orW1'])
            examples.append([
                bert.SingleInputExample(guid=guid,
                                        text_a=text_0,
                                        label=label),
                bert.SingleInputExample(guid=guid,
                                        text_a=text_1,
                                        label=label)])
        return examples


class DataLoadersAdvOriginalW(DataLoadersAdvOriginal):

    def create_examples(self, df):
        examples = []
        for _, line in df.iterrows():
            guid = line['#id']
            text_0 = line['warrant0']
            text_1 = line['warrant1']
            label = int(line['correctLabelW0orW1'])
            examples.append([
                bert.SingleInputExample(guid=guid,
                                        text_a=text_0,
                                        label=label),
                bert.SingleInputExample(guid=guid,
                                        text_a=text_1,
                                        label=label)])
        return examples


class DataLoadersAdvOriginalRW(DataLoadersAdvOriginal):

    def create_examples(self, df):
        examples = []
        for _, line in df.iterrows():
            guid = line['#id']
            text_a = line['reason']
            text_b_0 = line['warrant0']
            text_b_1 = line['warrant1']
            label = int(line['correctLabelW0orW1'])
            examples.append([
                bert.InputExample(guid=guid,
                                  text_a=text_a,
                                  text_b=text_b_0,
                                  label=label),
                bert.InputExample(guid=guid,
                                  text_a=text_a,
                                  text_b=text_b_1,
                                  label=label)])
        return examples


class DataLoadersAdvOriginalCW(DataLoadersAdvOriginal):

    def create_examples(self, df):
        examples = []
        for _, line in df.iterrows():
            guid = line['#id']
            text_a = line['claim']
            text_b_0 = line['warrant0']
            text_b_1 = line['warrant1']
            label = int(line['correctLabelW0orW1'])
            examples.append([
                bert.InputExample(guid=guid,
                                  text_a=text_a,
                                  text_b=text_b_0,
                                  label=label),
                bert.InputExample(guid=guid,
                                  text_a=text_a,
                                  text_b=text_b_1,
                                  label=label)])
        return examples


class DataLoadersWW(DataLoaders):

    def create_examples(self, df):
        examples = []
        for _, line in df.iterrows():
            guid = line['#id']
            text_0 = line['warrant0']
            text_1 = line['warrant1']
            label = int(line['correctLabelW0orW1'])
            examples.append([
                bert.SingleInputExample(guid=guid,
                                        text_a=text_0,
                                        label=label),
                bert.SingleInputExample(guid=guid,
                                        text_a=text_1,
                                        label=label)])
        return examples


class DataLoadersRW(DataLoaders):

    def create_examples(self, df):
        examples = []
        for _, line in df.iterrows():
            guid = line['#id']
            text_a = line['reason']
            text_b_0 = line['warrant0']
            text_b_1 = line['warrant1']
            label = int(line['correctLabelW0orW1'])
            examples.append([
                bert.InputExample(guid=guid,
                                  text_a=text_a,
                                  text_b=text_b_0,
                                  label=label),
                bert.InputExample(guid=guid,
                                  text_a=text_a,
                                  text_b=text_b_1,
                                  label=label)])
        return examples


class DataLoadersCW(DataLoaders):

    def create_examples(self, df):
        examples = []
        for _, line in df.iterrows():
            guid = line['#id']
            text_a = line['claim']
            text_b_0 = line['warrant0']
            text_b_1 = line['warrant1']
            label = int(line['correctLabelW0orW1'])
            examples.append([
                bert.InputExample(guid=guid,
                                  text_a=text_a,
                                  text_b=text_b_0,
                                  label=label),
                bert.InputExample(guid=guid,
                                  text_a=text_a,
                                  text_b=text_b_1,
                                  label=label)])
        return examples


#
# Models

class BERT(modeling.PreTrainedBertModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.bert = ppb.BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    @classmethod
    def from_args(cls, args):
        return cls.from_pretrained(args.bert_model)

    def forward(self, batch):
        input_ids_0, input_mask_0, segment_ids_0, \
        input_ids_1, input_mask_1, segment_ids_1, label_ids = \
            batch.tensors()

        _, pooled_output_0 = self.bert(
            input_ids_0, segment_ids_0, input_mask_0,
            output_all_encoded_layers=False)
        pooled_output_0 = self.dropout(pooled_output_0)

        _, pooled_output_1 = self.bert(
            input_ids_1, segment_ids_1, input_mask_1,
            output_all_encoded_layers=False)
        pooled_output_1 = self.dropout(pooled_output_1)

        logits_0 = self.classifier(pooled_output_0)
        logits_1 = self.classifier(pooled_output_1)

        logits = torch.cat([logits_0, logits_1], dim=1)

        loss = self.loss(logits.view(-1, self.num_labels), label_ids.view(-1))
        return loss, logits


class AttentionTrackingBERT(modeling.PreTrainedBertModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.bert = bert.AttentionTrackingBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    @classmethod
    def from_args(cls, args):
        return cls.from_pretrained(args.bert_model)

    def forward(self, batch):
        input_ids_0, input_mask_0, segment_ids_0, \
        input_ids_1, input_mask_1, segment_ids_1, label_ids = \
            batch.tensors()

        _, pooled_output_0 = self.bert(
            input_ids_0, segment_ids_0, input_mask_0,
            output_all_encoded_layers=False)
        pooled_output_0 = self.dropout(pooled_output_0)

        _, pooled_output_1 = self.bert(
            input_ids_1, segment_ids_1, input_mask_1,
            output_all_encoded_layers=False)
        pooled_output_1 = self.dropout(pooled_output_1)

        logits_0 = self.classifier(pooled_output_0)
        logits_1 = self.classifier(pooled_output_1)

        logits = torch.cat([logits_0, logits_1], dim=1)

        loss = self.loss(logits.view(-1, self.num_labels), label_ids.view(-1))
        return loss, logits
