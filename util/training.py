"""Functions for training."""
import os
import tqdm
import torch
import random
import glovar
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import pytorch_pretrained_bert as ppb


class Args:

    def __init__(self, experiment_name, bert_model='bert-large-uncased',
                 eval_batch_size=1, learning_rate=5e-5, num_train_epochs=3,
                 gradient_accumulation_steps=16, seed=42, local_rank=-1,
                 no_cuda=False, fp16=False, max_seq_length=128,
                 optimize_on_cpu=False, loss_scale=128, train_batch_size=8,
                 warmup_proportion=0.1, annealing_factor=0.5, use_bert=True,
                 **kwargs):
        self.experiment_name = experiment_name
        self.use_bert = use_bert
        self.bert_model = bert_model
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.local_rank = local_rank
        self.no_cuda = no_cuda
        self.fp16 = fp16
        self.seed = seed
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.optimize_on_cpu = optimize_on_cpu
        self.loss_scale = loss_scale
        self.warmup_proportion = warmup_proportion
        self.annealing_factor = annealing_factor
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def print(self):
        print('Args:')
        for key in sorted(self.__dict__.keys()):
            print('\t%s:\t%s' % (key, self.__dict__[key]))


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    for (name_opti, param_opti), (name_model, param_model) \
            in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            print("name_opti != name_model: {} {}".format(name_opti,
                                                          name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def determine_gradient_accumulation(args):
    if not args.use_bert:
        args.gradient_accumulation_steps = 1
    elif args.bert_model.startswith('bert-base'):     # limit of 8 per batch
        args.gradient_accumulation_steps = int(args.train_batch_size / 8)
    elif args.bert_model.startswith('bert-large'):  # limit of 1 per batch
        args.gradient_accumulation_steps = args.train_batch_size
    else:
        raise ValueError('Unexpected bert model:  %s' % args.bert_model)


def determine_train_batch_size(args):
    args.train_batch_size = int(args.train_batch_size
                                / args.gradient_accumulation_steps)


def set_optimizer_params_grad(named_params_optimizer, named_params_model,
                              test_nan=False):
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) \
            in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            print("name_opti != name_model: {} {}".format(name_opti,
                                                          name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(
                    param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


class Saver:

    def __init__(self, ckpt_dir):
        self.ckpt_dir = ckpt_dir

    def ckpt_path(self, name, module, is_best):
        return os.path.join(
            self.ckpt_dir,
            '%s_%s_%s' % (name, module, 'best' if is_best else 'latest'))

    def load(self, model, name, is_best=False, load_to_cpu=False,
             load_optimizer=True, replace_model=None, ignore_missing=False):
        model_path = self.ckpt_path(name, 'model', is_best)
        model_state_dict = self.get_state_dict(model_path, load_to_cpu)
        model_state_dict = self.replace_model_state(
            model_state_dict, replace_model)
        if ignore_missing:
            model_state_dict = self.drop_missing(model, model_state_dict)
        model.load_state_dict(model_state_dict)
        if load_optimizer:
            optim_path = self.ckpt_path(name, 'optim', is_best)
            optim_state_dict = self.get_state_dict(optim_path, load_to_cpu)
            model.optimizer.load_state_dict(optim_state_dict)

    @staticmethod
    def drop_missing(model, saved_state_dict):
        return {k: v for k, v in saved_state_dict.items()
                if k in model.state_dict().keys()}

    @staticmethod
    def replace_model_state(state_dict, replace):
        if replace is not None:
            for name, tensor in replace.items():
                state_dict[name] = tensor
        return state_dict

    @staticmethod
    def filter_optim_state_dict(state_dict, exclude):
        if exclude is not None:
            raise NotImplementedError  # TODO
        else:
            return state_dict

    @staticmethod
    def get_state_dict(path, load_to_cpu):
        if not torch.cuda.is_available() or load_to_cpu:
            return torch.load(path, map_location=lambda storage, loc: storage)
        else:
            return torch.load(path)

    def save(self, model, name, is_best, save_optim=False):
        model_path = self.ckpt_path(name, 'model', False)
        torch.save(model.state_dict(), model_path)
        if is_best:
            model_path = self.ckpt_path(name, 'model', True)
            torch.save(model.state_dict(), model_path)
        if save_optim:
            optim_path = self.ckpt_path(name, 'optim', is_best)
            torch.save(model.optimizer.state_dict(), optim_path)


class Batch:

    def to(self, device):
        raise NotImplementedError


class DataLoaders:

    def train(self, args):
        raise NotImplementedError

    def dev(self, args):
        raise NotImplementedError

    def test(self, args):
        raise NotImplementedError

    def test_adv(self, args):
        raise NotImplementedError

    def num_train_steps(self, args):
        if not self.n_training_points:
            raise ValueError('n_training_points not initialized.')
        return int(self.n_training_points
                   / args.train_batch_size
                   / args.gradient_accumulation_steps
                   * args.num_train_epochs)


def train(args, model, data_loaders):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                                        and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes distributed backend taking care of syncing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            print("16-bits training currently not supported in "
                  "distributed training")
            args.fp16 = False  # https://github.com/pytorch/pytorch/
                               # pull/13496
    print("device %s n_gpu %d distributed training %r"
          % (device, n_gpu, bool(args.local_rank != -1)))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: "
                         "{}, should be >= 1"
                         .format(args.gradient_accumulation_steps))

    # create dir for experiment checkpoints if it doesn't exist
    experiment_ckpt_dir = os.path.join(glovar.CKPT_DIR, args.experiment_name)
    if not os.path.exists(experiment_ckpt_dir):
        os.mkdir(experiment_ckpt_dir)

    # load data
    train_loader = data_loaders.train(args)
    dev_loader = data_loaders.dev(args)
    test_loader = data_loaders.test(args)
    num_train_steps = data_loaders.num_train_steps(args)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [
            (n, param.clone().detach().to('cpu').float().requires_grad_())
            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [
            (n, param.clone().detach().to('cpu').requires_grad_())
            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.use_bert:
        optimizer = ppb.BertAdam(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            warmup=args.warmup_proportion,
            t_total=t_total)
    else:
        optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)

    # training process
    global_step = 0
    saver = Saver(ckpt_dir=experiment_ckpt_dir)
    annealer = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='max', factor=args.annealing_factor)

    # Epochs
    model.train()
    dev_accs = []
    for _ in tqdm.trange(int(args.num_train_epochs), desc="Epoch"):
        train_accuracy = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm.tqdm(train_loader,
                                               desc="Iteration")):
            batch.to(device)
            loss, logits = model(batch)
            logits = logits.detach().cpu().numpy()
            tmp_train_accuracy = accuracy(
                logits, batch.label_ids.detach().cpu().numpy())
            train_accuracy += tmp_train_accuracy
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.fp16 and args.loss_scale != 1.0:
                # rescale loss for fp16 training
                loss = loss * args.loss_scale
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            nb_tr_examples += batch.label_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16 or args.optimize_on_cpu:
                    if args.fp16 and args.loss_scale != 1.0:
                        # scale down gradients for fp16 training
                        for param in model.parameters():
                            if param.grad is not None:
                                param.grad.data = param.grad.data / \
                                                  args.loss_scale
                    is_nan = set_optimizer_params_grad(
                        param_optimizer,
                        model.named_parameters(),
                        test_nan=True)
                    if is_nan:
                        print("FP16 TRAINING: Nan in gradients,"
                              "reducing loss scaling")
                        args.loss_scale = args.loss_scale / 2
                        model.zero_grad()
                        continue
                    optimizer.step()
                    copy_optimizer_params_to_model(model.named_parameters(),
                                                   param_optimizer)
                else:
                    optimizer.step()
                model.zero_grad()
                global_step += 1
        train_accuracy = train_accuracy / nb_tr_examples
        print('Accumulated training acc this epoch: %s' % train_accuracy)

        # Tune on the dev set
        dev_acc, _ = acc_and_preds(model, dev_loader, device)
        is_best = dev_acc > np.max(dev_accs) \
            if len(dev_accs) > 0 else True

        dev_accs.append(dev_acc)
        print('\nDev acc: %s' % dev_acc)
        print('Is best: %s' % is_best)

        # learning rate annealing
        annealer.step(dev_acc)

        # save params
        saver.save(model, args.experiment_name, is_best)

    # load the best model
    saver.load(
        model, args.experiment_name, is_best=True, load_optimizer=False)

    # evaluate best model on all sets
    train_acc, train_preds = acc_and_preds(model, train_loader, device)
    dev_acc, dev_preds = acc_and_preds(model, dev_loader, device)
    test_acc, test_preds = acc_and_preds(model, test_loader, device)

    accs = {
        'train': train_acc,
        'dev': dev_acc,
        'test': test_acc}
    preds = {
        'train': train_preds,
        'dev': dev_preds,
        'test': test_preds}

    return accs, preds


def acc_and_preds(model, data_loader, device):
    model.eval()
    cum_acc = 0.
    n_steps, n_examples = 0, 0
    preds = []
    softmax = nn.Softmax(dim=-1)
    with tqdm.tqdm(total=len(data_loader)) as pbar:
        for batch in data_loader:
            batch.to(device)
            with torch.no_grad():
                loss, logits = model(batch)
            n_examples += batch.label_ids.size(0)
            probs = softmax(logits).detach().cpu().numpy()
            _preds = logits.max(dim=1)[1].detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            label_ids = batch.label_ids.detach().cpu().numpy()
            tmp_acc = accuracy(logits, label_ids)
            cum_acc += tmp_acc
            n_steps += 1
            correct = _preds == label_ids
            for i in range(len(batch)):
                preds.append({
                    'id': batch.ids[i],
                    'prob0': probs[i][0],
                    'prob1': probs[i][1],
                    'pred': _preds[i],
                    'correct': correct[i]})
            pbar.update()

    acc = cum_acc / n_examples

    model.train()

    return acc, preds
