"""BiLSTM baselines."""
import torch
from torch import nn
from torch.utils import data as torch_data
import numpy as np
from util import training, text
from arct import data


#
# Data Preparation


class RNNSents:

    def __init__(self, sents, lens, rev_ix_sort):
        self.sents = sents
        self.lens = lens
        self.rev_ix_sort = rev_ix_sort


class CollateSentsForRNN:

    def __init__(self):
        self.vocab = data.vocab()
        self.tokenizer = text.tokenize
        self.pad_ix = 0

    def __call__(self, sents):
        sents = [self.tokenizer(sent) for sent in sents]
        sents = self.tokens_to_ixs(sents)
        lens = [len(sent) for sent in sents]
        lens = np.array(lens)
        self.pad(sents, max(lens))
        sents, lens, ix_sort = self.sort_by_len(sents, lens)
        sents = np.stack([np.array(s) for s in sents])
        rev_ix_sort = np.argsort(ix_sort)
        lens = torch.LongTensor(lens.copy())
        return RNNSents(sents, lens, rev_ix_sort)

    def pad(self, sents, max_len):
        for sent in sents:
            while len(sent) < max_len:
                sent.append(self.pad_ix)

    def tokens_to_ixs(self, sents):
        return [[self.vocab[tok] for tok in sent] for sent in sents]

    @staticmethod
    def sort_by_len(sents, lens):
        lens, ix_sort = np.sort(lens)[::-1], np.argsort(-lens)
        sents = list(np.array(sents)[ix_sort])
        return sents, lens, ix_sort


class Dataset(torch_data.Dataset):

    def __init__(self, data_points):
        self.data_points = data_points

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, item):
        return self.data_points.iloc[item]


class Batch(training.Batch):

    def __init__(self, ids, claims, reasons, warrant0s, warrant1s, label_ids):
        self.ids = ids
        self.claims = claims
        self.reasons = reasons
        self.warrant0s = warrant0s
        self.warrant1s = warrant1s
        self.label_ids = label_ids

    def __len__(self):
        return len(self.ids)

    def to(self, device):
        self.claims.sents = self.claims.sents.to(device)
        self.reasons.sents = self.reasons.sents.to(device)
        self.warrant0s.sents = self.warrant0s.sents.to(device)
        self.warrant1s.sents = self.warrant1s.sents.to(device)
        self.label_ids = self.label_ids.to(device)


class Collate:

    def __init__(self):
        self.vocab = data.vocab()
        self.collate_for_rnn = CollateSentsForRNN()

    def __call__(self, items):
        ids = [x['#id'] for x in items]
        claims = self.collate_for_rnn([x['claim'] for x in items])
        reasons = self.collate_for_rnn([x['reason'] for x in items])
        warrant0s = self.collate_for_rnn([x['warrant0'] for x in items])
        warrant1s = self.collate_for_rnn([x['warrant1'] for x in items])
        label_ids = torch.LongTensor([x['correctLabelW0orW1'] for x in items])

        claims.sents = torch.LongTensor(claims.sents)
        reasons.sents = torch.LongTensor(reasons.sents)
        warrant0s.sents = torch.LongTensor(warrant0s.sents)
        warrant1s.sents = torch.LongTensor(warrant1s.sents)

        return Batch(
            ids=ids,
            claims=claims,
            reasons=reasons,
            warrant0s=warrant0s,
            warrant1s=warrant1s,
            label_ids=label_ids)

    def pad(self, ixs):
        lens = [len(s) for s in ixs]
        max_len = max(lens)
        for s in ixs:
            while len(s) < max_len:
                s.append(0)

    def tokenize_and_lookup(self, sents):
        sents = [text.tokenize(s) for s in sents]
        ixs = [[self.vocab[t] for t in s] for s in sents]
        lens = [len(s) for s in sents]
        self.pad(ixs)
        return torch.LongTensor(ixs), torch.LongTensor(lens)


class DataLoaders(training.DataLoaders):

    def __init__(self, args=None):
        self.vocab = data.vocab()
        self.n_training_points = None

    def train(self, args):
        data_points = data.load('train')
        self.n_training_points = len(data_points)
        return self.get_data_loader(data_points, args)

    def dev(self, args):
        return self.get_data_loader(data.load('dev'), args)

    def test(self, args):
        return self.get_data_loader(data.load('test'), args)

    def test_adv(self, args):
        return self.get_data_loader(data.load('test-adv'), args)

    def get_data_loader(self, data_points, args):
        dataset = Dataset(data_points)
        sampler = torch_data.RandomSampler(dataset)
        return torch_data.DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=args.train_batch_size,
            collate_fn=Collate())


class DataLoadersAdv(DataLoaders):

    def train(self, args):
        data_points = data.load('train-adv')
        self.n_training_points = len(data_points)
        return self.get_data_loader(data_points, args)

    def dev(self, args):
        data_points = data.load('dev-adv')
        return self.get_data_loader(data_points, args)

    def test(self, args):
        data_points = data.load('test-adv')
        return self.get_data_loader(data_points, args)


#
# Models


class Encoder(nn.Module):

    def __init__(self, hidden_size, embed_size=300):
        super(Encoder, self).__init__()
        self.encoder = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True)

    def forward(self, sents):
        sents.sents = sents.sents.permute([1, 0, 2])
        packed = nn.utils.rnn.pack_padded_sequence(sents.sents, sents.lens)
        outputs = self.encoder(packed)[0]
        outputs = nn.utils.rnn.pad_packed_sequence(outputs)[0]

        outputs = outputs.permute([1, 0, 2])
        if torch.cuda.is_available():
            outputs = outputs[torch.LongTensor(sents.rev_ix_sort).cuda()]
        else:
            outputs = outputs[torch.LongTensor(sents.rev_ix_sort)]

        return outputs


class BiLSTM(nn.Module):

    def __init__(self, args):
        super().__init__()
        args.hidden_size = int(args.hidden_size)
        embeds = torch.from_numpy(data.glove())
        self.embeddings = nn.Embedding(embeds.shape[0], embeds.shape[1])
        self.embeddings.weight = nn.Parameter(
            embeds, requires_grad=args.tune_embeds)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.encoder = Encoder(args.hidden_size)
        self.classify = nn.Linear(args.hidden_size * 6, 1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        # embedding lookup
        batch.reasons.sents = self.embeddings(batch.reasons.sents)
        batch.claims.sents = self.embeddings(batch.claims.sents)
        batch.warrant0s.sents = self.embeddings(batch.warrant0s.sents)
        batch.warrant1s.sents = self.embeddings(batch.warrant1s.sents)

        # encoding
        reasons = self.encoder(batch.reasons)
        claims = self.encoder(batch.claims)
        warrant0s = self.encoder(batch.warrant0s)
        warrant1s = self.encoder(batch.warrant1s)

        # max pooling
        reasons = torch.max(reasons, dim=1)[0]
        claims = torch.max(claims, dim=1)[0]
        warrant0s = torch.max(warrant0s, dim=1)[0]
        warrant1s = torch.max(warrant1s, dim=1)[0]

        # gather for classification
        input0 = torch.cat([reasons, claims, warrant0s], dim=1)
        input1 = torch.cat([reasons, claims, warrant1s], dim=1)

        # regularization
        input0 = self.dropout(input0)
        input1 = self.dropout(input1)

        # classification
        logits0 = self.classify(input0)
        logits1 = self.classify(input1)

        # cat logits for softmax
        logits = torch.cat([logits0, logits1], dim=1)

        # calculate loss
        loss = self.loss(logits, batch.label_ids)

        return loss, logits


class BiLSTM_WW(nn.Module):

    def __init__(self, args):
        super().__init__()
        embeds = torch.from_numpy(data.glove())
        self.embeddings = nn.Embedding(embeds.shape[0], embeds.shape[1])
        self.embeddings.weight = nn.Parameter(
            embeds, requires_grad=args.tune_embeds)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.encoder = Encoder(args.hidden_size)
        self.classify = nn.Linear(args.hidden_size * 2, 1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        # embedding lookup
        batch.warrant0s.sents = self.embeddings(batch.warrant0s.sents)
        batch.warrant1s.sents = self.embeddings(batch.warrant1s.sents)

        # encoding
        warrant0s = self.encoder(batch.warrant0s)
        warrant1s = self.encoder(batch.warrant1s)

        # max pooling
        warrant0s = torch.max(warrant0s, dim=1)[0]
        warrant1s = torch.max(warrant1s, dim=1)[0]

        # gather for classification
        input0 = warrant0s
        input1 = warrant1s

        # regularization
        input0 = self.dropout(input0)
        input1 = self.dropout(input1)

        # classification
        logits0 = self.classify(input0)
        logits1 = self.classify(input1)

        # cat logits for softmax
        logits = torch.cat([logits0, logits1], dim=1)

        # calculate loss
        loss = self.loss(logits, batch.label_ids)

        return loss, logits


class BiLSTM_RW(nn.Module):

    def __init__(self, args):
        super().__init__()
        embeds = torch.from_numpy(data.glove())
        self.embeddings = nn.Embedding(embeds.shape[0], embeds.shape[1])
        self.embeddings.weight = nn.Parameter(
            embeds, requires_grad=args.tune_embeds)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.encoder = Encoder(args.hidden_size)
        self.classify = nn.Linear(args.hidden_size * 4, 1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        # embedding lookup
        batch.reasons.sents = self.embeddings(batch.reasons.sents)
        batch.warrant0s.sents = self.embeddings(batch.warrant0s.sents)
        batch.warrant1s.sents = self.embeddings(batch.warrant1s.sents)

        # encoding
        reasons = self.encoder(batch.reasons)
        warrant0s = self.encoder(batch.warrant0s)
        warrant1s = self.encoder(batch.warrant1s)

        # max pooling
        reasons = torch.max(reasons, dim=1)[0]
        warrant0s = torch.max(warrant0s, dim=1)[0]
        warrant1s = torch.max(warrant1s, dim=1)[0]

        # gather for classification
        input0 = torch.cat([reasons, warrant0s], dim=1)
        input1 = torch.cat([reasons, warrant1s], dim=1)

        # regularization
        input0 = self.dropout(input0)
        input1 = self.dropout(input1)

        # classification
        logits0 = self.classify(input0)
        logits1 = self.classify(input1)

        # cat logits for softmax
        logits = torch.cat([logits0, logits1], dim=1)

        # calculate loss
        loss = self.loss(logits, batch.label_ids)

        return loss, logits


class BiLSTM_CW(nn.Module):

    def __init__(self, args):
        super().__init__()
        embeds = torch.from_numpy(data.glove())
        self.embeddings = nn.Embedding(embeds.shape[0], embeds.shape[1])
        self.embeddings.weight = nn.Parameter(
            embeds, requires_grad=args.tune_embeds)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.encoder = Encoder(args.hidden_size)
        self.classify = nn.Linear(args.hidden_size * 4, 1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        # embedding lookup
        batch.claims.sents = self.embeddings(batch.claims.sents)
        batch.warrant0s.sents = self.embeddings(batch.warrant0s.sents)
        batch.warrant1s.sents = self.embeddings(batch.warrant1s.sents)

        # encoding
        claims = self.encoder(batch.claims)
        warrant0s = self.encoder(batch.warrant0s)
        warrant1s = self.encoder(batch.warrant1s)

        # max pooling
        claims = torch.max(claims, dim=1)[0]
        warrant0s = torch.max(warrant0s, dim=1)[0]
        warrant1s = torch.max(warrant1s, dim=1)[0]

        # gather for classification
        input0 = torch.cat([claims, warrant0s], dim=1)
        input1 = torch.cat([claims, warrant1s], dim=1)

        # regularization
        input0 = self.dropout(input0)
        input1 = self.dropout(input1)

        # classification
        logits0 = self.classify(input0)
        logits1 = self.classify(input1)

        # cat logits for softmax
        logits = torch.cat([logits0, logits1], dim=1)

        # calculate loss
        loss = self.loss(logits, batch.label_ids)

        return loss, logits
