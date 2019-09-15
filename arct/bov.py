"""Bag of Vectors."""
import torch
from torch import nn
from torch.utils import data as torch_data
from util import training, text
from arct import data
import numpy as np


#
# Utilities


def vectorize(sent):
    tokens = text.tokenize(sent)
    vocab = data.vocab()
    ixs = [vocab[t] for t in tokens]
    embeddings = data.glove()
    _vectors = [embeddings[ix].reshape(1, -1) for ix in ixs]
    return np.concatenate(_vectors, axis=0).sum(axis=0)


def vectors(x):
    claim = vectorize(x['claim'])
    reason = vectorize(x['reason'])
    argument = np.concatenate([claim.reshape(1, -1),
                               reason.reshape(1, -1)], axis=0).sum(axis=0)
    warrant0 = vectorize(x['warrant0'])
    warrant1 = vectorize(x['warrant1'])
    return claim, reason, argument, warrant0, warrant1


#
# Data Preparation


class Dataset(torch_data.Dataset):

    def __init__(self, data_points):
        self.data_points = data_points

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, item):
        return self.data_points.iloc[item]


class Batch(training.Batch):

    def __init__(self, ids, claims, reasons, warrant0s, warrant1s, label_ids,
                 claim_lens, reason_lens, warrant0_lens, warrant1_lens):
        self.ids = ids
        self.claims = claims
        self.reasons = reasons
        self.warrant0s = warrant0s
        self.warrant1s = warrant1s
        self.label_ids = label_ids
        self.claim_lens = claim_lens
        self.reason_lens = reason_lens
        self.warrant0_lens = warrant0_lens
        self.warrant1_lens = warrant1_lens

    def __len__(self):
        return len(self.ids)

    def to(self, device):
        self.claims = self.claims.to(device)
        self.reasons = self.reasons.to(device)
        self.warrant0s = self.warrant0s.to(device)
        self.warrant1s = self.warrant1s.to(device)
        self.label_ids = self.label_ids.to(device)


class Collate:

    def __init__(self):
        self.vocab = data.vocab()

    def __call__(self, items):
        ids = [x['#id'] for x in items]
        claims, claim_lens = self.tokenize_and_lookup(
            [x['claim'] for x in items])
        reasons, reason_lens = self.tokenize_and_lookup(
            [x['reason'] for x in items])
        warrant0s, warrant0_lens = self.tokenize_and_lookup(
            [x['warrant0'] for x in items])
        warrant1s, warrant1_lens = self.tokenize_and_lookup(
            [x['warrant1'] for x in items])
        label_ids = torch.LongTensor([x['correctLabelW0orW1'] for x in items])
        return Batch(
            ids=ids,
            claims=claims,
            reasons=reasons,
            warrant0s=warrant0s,
            warrant1s=warrant1s,
            label_ids=label_ids,
            claim_lens=claim_lens,
            reason_lens=reason_lens,
            warrant0_lens=warrant0_lens,
            warrant1_lens=warrant1_lens)

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
        data_points = data.load('train-original')
        self.n_training_points = len(data_points)
        return self.get_data_loader(data_points, args)

    def dev(self, args):
        return self.get_data_loader(data.load('dev-original'), args)

    def test(self, args):
        return self.get_data_loader(data.load('test-original'), args)

    @staticmethod
    def get_data_loader(data_points, args):
        dataset = Dataset(data_points)
        sampler = torch_data.RandomSampler(dataset)
        return torch_data.DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=args.train_batch_size,
            collate_fn=Collate())


class DataLoadersSwappedTrain(DataLoaders):

    def train(self, args):
        data_points = data.load('train-adv-swapped')
        self.n_training_points = len(data_points)
        return self.get_data_loader(data_points, args)

    def dev(self, args):
        data_points = data.load('dev-original')
        return self.get_data_loader(data_points, args)

    def test(self, args):
        data_points = data.load('test-original')
        return self.get_data_loader(data_points, args)


class DataLoadersAdvOriginal(DataLoaders):

    def train(self, args):
        data_points = data.load('train-adv-swapped')
        self.n_training_points = len(data_points)
        return self.get_data_loader(data_points, args)

    def dev(self, args):
        data_points = data.load('dev-adv-negated')
        return self.get_data_loader(data_points, args)

    def test(self, args):
        data_points = data.load('test-adv-negated')
        return self.get_data_loader(data_points, args)


class DataLoadersAdvSwapped(DataLoaders):

    def train(self, args):
        data_points = data.load('train-adv-swapped')
        self.n_training_points = len(data_points)
        return self.get_data_loader(data_points, args)

    def dev(self, args):
        data_points = data.load('dev-adv-swapped')
        return self.get_data_loader(data_points, args)

    def test(self, args):
        data_points = data.load('test-adv-swapped')
        return self.get_data_loader(data_points, args)


class DataLoadersAdvNegated(DataLoaders):

    def train(self, args):
        data_points = data.load('train-adv-negated')
        self.n_training_points = len(data_points)
        return self.get_data_loader(data_points, args)

    def dev(self, args):
        data_points = data.load('dev-adv-negated')
        return self.get_data_loader(data_points, args)

    def test(self, args):
        data_points = data.load('test-adv-negated')
        return self.get_data_loader(data_points, args)


#
# Models


class BOV(nn.Module):

    def __init__(self, args):
        super().__init__()
        embeds = torch.from_numpy(data.glove())
        self.embeddings = nn.Embedding(embeds.shape[0], embeds.shape[1])
        self.embeddings.weight = nn.Parameter(
            embeds, requires_grad=args.tune_embeds)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classify = nn.Linear(600, 1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        # batch.{sent} is vectors of shape batch x max_len x dim (300)
        # batch tensors should be put on device in training script prior to here

        # embedding lookup
        reasons = self.embeddings(batch.reasons)
        claims = self.embeddings(batch.claims)
        warrant0s = self.embeddings(batch.warrant0s)
        warrant1s = self.embeddings(batch.warrant1s)

        # composition
        args = torch.cat([reasons, claims], dim=1)

        # reduction
        args = torch.max(args, dim=1)[0]
        warrant0s = torch.max(warrant0s, dim=1)[0]
        warrant1s = torch.max(warrant1s, dim=1)[0]

        # gather for classification
        input0 = torch.cat([args, warrant0s], dim=1)
        input1 = torch.cat([args, warrant1s], dim=1)

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


class BOV_RW(nn.Module):

    def __init__(self, args):
        super().__init__()
        embeds = torch.from_numpy(data.glove())
        self.embeddings = nn.Embedding(embeds.shape[0], embeds.shape[1])
        self.embeddings.weight = nn.Parameter(
            embeds, requires_grad=args.tune_embeds)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classify = nn.Linear(600, 1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        # batch.{sent} is vectors of shape batch x max_len x dim (300)
        # batch tensors should be put on device in training script prior to here

        # embedding lookup
        reasons = self.embeddings(batch.reasons)
        warrant0s = self.embeddings(batch.warrant0s)
        warrant1s = self.embeddings(batch.warrant1s)

        # reduction
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


class BOV_CW(nn.Module):

    def __init__(self, args):
        super().__init__()
        embeds = torch.from_numpy(data.glove())
        self.embeddings = nn.Embedding(embeds.shape[0], embeds.shape[1])
        self.embeddings.weight = nn.Parameter(
            embeds, requires_grad=args.tune_embeds)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classify = nn.Linear(600, 1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        # batch.{sent} is vectors of shape batch x max_len x dim (300)
        # batch tensors should be put on device in training script prior to here

        # embedding lookup
        claims = self.embeddings(batch.claims)
        warrant0s = self.embeddings(batch.warrant0s)
        warrant1s = self.embeddings(batch.warrant1s)

        # reduction
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


class BOV_CR(nn.Module):

    def __init__(self, args):
        super().__init__()
        embeds = torch.from_numpy(data.glove())
        self.embeddings = nn.Embedding(embeds.shape[0], embeds.shape[1])
        self.embeddings.weight = nn.Parameter(
            embeds, requires_grad=args.tune_embeds)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classify = nn.Linear(600, 2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        # batch.{sent} is vectors of shape batch x max_len x dim (300)
        # batch tensors should be put on device in training script prior to here

        # embedding lookup
        claims = self.embeddings(batch.claims)
        reasons = self.embeddings(batch.reasons)

        # reduction
        claims = torch.max(claims, dim=1)[0]
        reasons = torch.max(reasons, dim=1)[0]

        # gather for classification
        input = torch.cat([claims, reasons], dim=1)

        # regularization
        input = self.dropout(input)

        # classification
        logits = self.classify(input)

        # calculate loss
        loss = self.loss(logits, batch.label_ids)

        return loss, logits


class BOV_W(nn.Module):

    def __init__(self, args):
        super().__init__()
        embeds = torch.from_numpy(data.glove())
        self.embeddings = nn.Embedding(embeds.shape[0], embeds.shape[1])
        self.embeddings.weight = nn.Parameter(
            embeds, requires_grad=args.tune_embeds)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classify = nn.Linear(300, 1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        # batch.{sent} is vectors of shape batch x max_len x dim (300)
        # batch tensors should be put on device in training script prior to here

        # embedding lookup
        warrant0s = self.embeddings(batch.warrant0s)
        warrant1s = self.embeddings(batch.warrant1s)

        # reduction
        warrant0s = torch.max(warrant0s, dim=1)[0]
        warrant1s = torch.max(warrant1s, dim=1)[0]

        # regularization
        input0 = self.dropout(warrant0s)
        input1 = self.dropout(warrant1s)

        # classification
        logits0 = self.classify(input0)
        logits1 = self.classify(input1)

        # cat logits for softmax
        logits = torch.cat([logits0, logits1], dim=1)

        # calculate loss
        loss = self.loss(logits, batch.label_ids)

        return loss, logits
