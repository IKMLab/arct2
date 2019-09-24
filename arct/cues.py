import math

from nltk import tokenize as tk
import numpy as np
from tqdm import tqdm_notebook as tqdm


def tokenize(sent):
    return [t.lower() for t in tk.word_tokenize(sent)]


#
# PMI Calculations


class PMI:
    """Default case - looks at the words in all the sentences."""

    def __init__(self, smoothing=100):
        self.smoothing = smoothing

    def __call__(self, df, view='correct', k=10):
        # get the token set
        tokens = self.get_token_set(df)

        # generate all counts of interest
        counts = {'0': {}, '1': {}, '0_correct': {}, '1_correct': {},
                  'correct': {}}
        for token in tokens:
            counts['0'][token] = 0
            counts['1'][token] = 0
            counts['0_correct'][token] = 0
            counts['1_correct'][token] = 0
            counts['correct'][token] = 0
        with tqdm(total=len(df), desc='counts') as pbar:
            for _, x in df.iterrows():
                tokens0, tokens1 = self.get_tokens(x)
                for token in tokens0:
                    counts['0'][token] += 1
                    if not x.correctLabelW0orW1:
                        counts['0_correct'][token] += 1
                        counts['correct'][token] += 1
                for token in tokens1:
                    counts['1'][token] += 1
                    if x.correctLabelW0orW1:
                        counts['1_correct'][token] += 1
                        counts['correct'][token] += 1
                pbar.update()

        # calculate PMI
        pmi = {'0': {}, '1': {}, 'correct': {}}
        with tqdm(total=len(tokens), desc='pmi') as pbar:
            for token in tokens:
                pmi['0'][token] = self.pmi(
                    n_class=counts['0_correct'][token],
                    n_total=counts['0'][token])
                pmi['1'][token] = self.pmi(
                    n_class=counts['1_correct'][token],
                    n_total=counts['1'][token])
                pmi['correct'][token] = self.pmi(
                    n_class=counts['correct'][token],
                    n_total=counts['0'][token] + counts['1'][token])
                pbar.update()

        # report results
        keys = ['0', '1', 'correct']
        if view:
            keys = [view]
        for key in keys:
            p = self.sort_pmi(pmi, key, tokens)
            print('-' * 8)
            print('PMI for label: %s' % key)
            vals = [pmi[key][t] for t in tokens]
            print('-' * 8)
            print('Min:      %s' % np.min(vals))
            print('Mean:     %s' % np.mean(vals))
            print('Std:      %s' % np.std(vals))
            print('Median:   %s' % np.median(vals))
            print('Max:      %s' % np.max(vals))
            print('Sum (>0): %s' % np.sum([v for v in vals if v > 0]))
            print('-' * 8)
            for i in range(k):
                print('%s\t%6.5f\t%s' % (i+1, p[i]['pmi'], p[i]['token']))

        return counts, pmi

    def get_tokens(self, x):
        claim = tokenize(x.claim)
        reason = tokenize(x.reason)
        warrant0 = tokenize(x.warrant0)
        warrant1 = tokenize(x.warrant1)
        tokens0 = list(set(claim + reason + warrant0))
        tokens1 = list(set(claim + reason + warrant1))
        return tokens0, tokens1

    def get_token_set(self, df):
        tokens = []
        for sent in ['claim', 'reason', 'warrant0', 'warrant1']:
            for x in df[sent].values:
                tokens += tokenize(x)
        return list(set(tokens))

    def pmi(self, n_class, n_total):
        return math.log((2 * n_class + self.smoothing) /
                        (n_total + self.smoothing))

    def sort_pmi(self, pmi, key, tokens):
        return list(reversed(sorted(
            [{'token': token, 'pmi': pmi[key][token]} for token in tokens],
            key=lambda x: x['pmi'])))


class PMIw(PMI):
    """Just looks at tokens in the warrants."""

    def get_tokens(self, x):
        warrant0 = list(set(tokenize(x.warrant0)))
        warrant1 = list(set(tokenize(x.warrant1)))
        return warrant0, warrant1


class PMIc(PMI):
    """Just looks at tokens in the claims."""

    def get_tokens(self, x):
        tokens = list(set(tokenize(x.claim)))
        if x.correctLabelW0orW1:  # 1 is the correct label
            return tokens, tokens
        else:  # 0 is the correct label
            return tokens, tokens


class PMIcw(PMI):

    def get_tokens(self, x):
        claim = list(set(tokenize(x.claim)))
        warrant0 = list(set(tokenize(x.warrant0)))
        warrant1 = list(set(tokenize(x.warrant1)))
        tokens0 = list(set(claim + warrant0))
        tokens1 = list(set(claim + warrant1))
        return tokens0, tokens1


#
# Heuristics


class Heuristic:
    """For calculating productivity and coverage."""

    def __call__(self, df):
        n = len(df)
        applicable = 0
        productive = 0
        with tqdm(total=len(df), desc='data point') as pbar:
            for _, x in df.iterrows():
                eval0, eval1 = self.assess(x)
                applicable += eval0 or eval1
                label = x.correctLabelW0orW1
                if not label and eval0:
                    productive += 1
                elif label and eval1:
                    productive += 1
                pbar.update()
        productivity = round((productive + 1) / (applicable + 1), 3)
        coverage = round((applicable + 1) / (n + 1), 3)
        print('-' * 8)
        print('Productivity: %s' % productivity)
        print('Coverage:     %s' % coverage)

    def assess(self, x):
        # returns eval0, eval1 - i.e. is the heuristic found in position 1 or 2
        raise NotImplementedError


class TokenInClaim(Heuristic):

    def __init__(self, token):
        self.token = token

    def assess(self, x):
        tokens = set(tokenize(x.claim))
        present = self.token in tokens
        return present, present


class TokenInClaimAndWarrant(Heuristic):

    def __init__(self, token):
        self.token = token

    def assess(self, x):
        claim = set(tokenize(x.claim))
        warrant0 = set(tokenize(x.warrant0))
        warrant1 = set(tokenize(x.warrant1))
        tokens0 = claim.intersection(warrant0)
        tokens1 = claim.intersection(warrant1)
        eval0 = self.token in tokens0
        eval1 = self.token in tokens1
        return eval0, eval1
