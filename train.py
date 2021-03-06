from __future__ import annotations

import argparse

import json
import math
import os
import typing
import time

import pandas as pd
from sklearn.preprocessing import minmax_scale
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from modelling import Embedder
from utils import read_one_hot_encodings, to_dense


class Instance(typing.TypedDict):
    one: torch.Tensor
    two: torch.Tensor
    sims: torch.Tensor


class Dataset(torch.utils.data.Dataset):
    def __init__(self, similarities_filename: str, ohe: dict[str, list[int]], exclude: list[str] = None):
        df = pd.read_csv(similarities_filename, sep='\t')
        df = df.set_index(['Entity 1', 'Entity 2'])

        if exclude is not None:
            df = df.drop(columns=exclude)

        df.iloc[:] = minmax_scale(df)

        self.data = df

        self.n_similarities = df.shape[1]

        self.ohe = ohe

        # The size of the one hot encoding is derived from the highest index
        # that is used in those encodings. Notice that we add 1 because the
        # indices are 0-based (thus, if the maximum number ever seen is 9, we
        # need vectors of dimension 10 to accommodate the encodings).
        self.n_concepts = max(i for indices in ohe.values() for i in indices) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Instance:
        name1, name2 = self.data.index[idx]

        return {
            'one': to_dense(self.ohe[name1], self.n_concepts),
            'two': to_dense(self.ohe[name2], self.n_concepts),
            'sims': torch.Tensor(self.data.iloc[idx]),
        }


def save_config(args, **kwargs):
    to_save = vars(args).copy()

    del to_save['dirname']

    to_save.update(kwargs)

    with open(os.path.join(args.dirname, 'config.json'), 'w') as f:
        json.dump(to_save, f)


def save_weights(filename: str, embedder: Embedder):
    weights = {
        k: v.to('cpu')
        for k, v in embedder.state_dict().items()
    }

    with open(filename, 'wb') as f:
        torch.save(weights, f)


@torch.no_grad()
def evaluate(
    model: Embedder,
    data: typing.Iterable[Instance],
    device: str,
    loss_function: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    n_similarities: int,
) -> float:
    model.eval()  # Place the model in evaluation mode

    total_loss = 0.0
    total_instances = 0

    for batch in tqdm(data, leave=False):
        output = model(
            one=batch['one'].to(device),
            two=batch['two'].to(device)
        )

        # Notice that here, contrary to what happens in training, we divide the
        # loss only by the number of target values. This is because we want to
        # accumulate the loss over all instances in all the eval batches, and
        # divide only by the total number of instances seen
        loss = loss_function(
            output,
            batch['sims'].to(device)
        ) / n_similarities

        total_loss += loss.item()
        total_instances += batch['one'].shape[0]

    model.train()  # Back to train mode

    return total_loss / total_instances


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Train the embedding model on a dataset, with the specified parameters.'
    )

    parser.add_argument(
        '-d', '--dirname',
        help='The directory where the output of training is stored. This is relative to the '
             '`outputs/` directory. It must not exist already. If not provided, one will be '
             'created named after the current timestamp.'
    )

    parser.add_argument(
        'similarities', metavar='SIMILARITIES',
        help='The filename pointing at the similarity values, calculated with `dataset.py`'
    )

    parser.add_argument(
        'encodings', metavar='ENCODINGS',
        help='The filename pointing at the one-hot encoding of the entities being compared.'
    )

    parser.add_argument(
        '-l', '--embedding-layers', type=int, default=4,
        help='The number of embedding layers used to embed an entity into a fixed-sized vector.'
    )

    parser.add_argument(
        '-m', '--mixer-layers', type=int, default=0,
        help='The number of mixer layers used to manipulate the concatenation of the two '
             'entity embeddings produced by the embedding layers.'
    )

    parser.add_argument(
        '-N', '--embedding-dimension', type=int, default=32,
        help='The number of nodes in each embedding layer. The number of nodes in the mixer '
             'layers is always double this.'
    )

    parser.add_argument(
        '-P', '--dropout', type=float, default=0.5,
        help='The dropout rate used to regularize the model.'
    )

    parser.add_argument(
        '-b', '--batch-size', type=int, default=4000,
        help='The number of instance to include in each mini-batch.'
    )

    parser.add_argument(
        '-e', '--eval-each', type=int, default=25,
        help='The number of training steps in between evaluation steps.'
    )

    parser.add_argument(
        '-S', '--train-split', type=float, default=0.95,
        help='The fraction of the dataset used for training (the rest is used for evaluation).'
    )

    parser.add_argument(
        '-r', '--learning-rate', type=float, default=1e-3,
        help='The learning rate of the training phase. The learning rate is adjusted during the '
             'whole session. See the `--scheduler-decay` flag for more information.'
    )

    parser.add_argument(
        '-c', '--scheduler-decay', default='linear',
        choices='none linear exponential'.split(),
        help='The type of adjustment to perform on the learning rate throughout the session. '
             '"none" means that the learning rate is kept constant; "linear" means it decreases '
             'linearly from the maximum value to a minimum (specified with the `--decay-param` '
             'flag); "exponential" means it is multiplied by a constant (specified with the '
             '`--decay-param` flag) each whole epoch.'
    )

    parser.add_argument(
        '-p', '--decay-param', type=float,
        help='The decay parameter of the scheduler. See the `--scheduler-decay` flag. If the '
             'scheuler decays linearly, the parameter is 0; if the decay is exponential, the '
             'parameter is a multiplication of 0.9 per epoch'
    )

    parser.add_argument(
        '-W', '--warmup', type=int, default=100,
        help='The number of training steps during which the learning rate grows linearly from 0 '
             'to the maximum value.'
    )

    parser.add_argument(
        '-n', '--epochs', type=int, default=20,
        help='The number of epochs the session should run through.'
    )

    parser.add_argument(
        '-w', '--weights-filename', default='model-weights.pt',
        help='The filename where the weights of the best model weights are saved.'
    )

    parser.add_argument(
        '-s', '--seed', type=int, default=42,
        help='The seed used to initialize the randomization processes.'
    )

    parser.add_argument(
        '-D', '--device', default='cpu',
        help='The device where the model will operate on. Defaults to cpu. Must be a string that '
             'torch recognizes as a valid device.'
    )

    parser.add_argument(
        '-x', '--exclude', nargs='+', default=[],
        help='If you want to exclude certain similarity values from the dataset, you can do it '
             'using this flag. Mutliple options are allowed, which must be separated by spaces. '
             'Note that if the similarity name has a space in it, it must be quoted. That is, you '
             'must take into account your shell parsing mechanisms.'
    )

    args = parser.parse_args()

    if args.dirname is None:
        args.dirname = time.strftime('%Y%m%d%H%M%S')

    args.dirname = os.path.join('outputs', args.dirname)

    args.weights_filename = os.path.join(args.dirname, args.weights_filename)

    if not args.decay_param:
        if args.scheduler_decay == 'linear':
            args.decay_param = 0.0
        elif args.scheduler_decay == 'exponential':
            args.decay_param = 0.9

    return args


def make_scheduler(optimizer: torch.optim.Optimizer, args, total_steps: int, steps_per_epoch: int):
    if args.scheduler_decay == 'none':
        return make_constant_scheduler(optimizer, args.warmup)

    if args.scheduler_decay == 'linear':
        return make_linear_scheduler(optimizer, args.warmup, args.decay_param, total_steps)

    if args.scheduler_decay == 'exponential':
        return make_exponential_decay(optimizer, args.warmup, args.decay_param, steps_per_epoch)


def make_constant_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int):
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / warmup_steps
        else:
            return 1

    return LambdaLR(optimizer, lr_lambda, -1)


def make_linear_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, target: float, total_steps: int):
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / warmup_steps
        else:
            # Math backs this section up. When `current_step == warmup_steps`
            # this results in 1, and when `current_step == total_steps` this
            # results in `target`. And it goes linearly from one to the other

            x = current_step - (current_step - warmup_steps) * target

            return (total_steps - x) / (total_steps - warmup_steps)

    return LambdaLR(optimizer, lr_lambda, -1)


def make_exponential_decay(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    decay_per_epoch: float,
    steps_per_epoch: int
):
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / warmup_steps
        else:
            # Math backs this section up. When `current_step == warmup_steps`
            # this results in 1, and when `current_step - warmup_steps ==
            # steps_per_epoch` this results in `decay_per_epoch`. And it goes
            # exponentially from one to the other

            return decay_per_epoch ** ((current_step - warmup_steps) / steps_per_epoch)

    return LambdaLR(optimizer, lr_lambda, -1)


def main():
    # Setup
    args = get_arguments()
    os.mkdir(args.dirname)
    torch.manual_seed(args.seed)

    # Read the data
    encodings = read_one_hot_encodings(args.encodings)
    dataset = Dataset(args.similarities, encodings, args.exclude)

    # Initialize the model
    embedder = Embedder(
        n_concepts=dataset.n_concepts,
        embedding_dimension=args.embedding_dimension,
        embedding_layers=args.embedding_layers,
        mixer_layers=args.mixer_layers,
        output_dimension=dataset.n_similarities,
        dropout=args.dropout,
    )
    embedder.to(args.device)
    embedder.train()

    # Split the data
    train_size = int(args.train_split * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(
        dataset,
        [train_size, eval_size],
        generator=None
    )

    train_dataloader = DataLoader(train_dataset, args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, args.batch_size)

    # Compute some training parameters based on the configuration
    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    n_steps = args.epochs * steps_per_epoch

    # Save the parameters
    save_config(
        args,
        steps_per_epoch=steps_per_epoch,
        n_concepts=dataset.n_concepts,
        n_similarities=dataset.n_similarities,
    )

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(embedder.parameters(), lr=args.learning_rate)
    scheduler = make_scheduler(optimizer, args, n_steps, steps_per_epoch)

    # Create the loss function. Notice that this sums over the errors over he
    # various output dimensions and batch sizes. To get a proper loss we must
    # divide by the number of instances in the batch as well as the number of
    # target dimensions
    loss_function = torch.nn.MSELoss(reduction='sum')

    # Prepare to write to tensorboard
    writer = SummaryWriter(args.dirname)

    # Save the best weights so far, measures during the evaluation phase
    best_eval_loss = float('inf')
    steps_until_evaluation = args.eval_each

    # Keep track of progress
    progress_bar = tqdm(total=n_steps, unit='steps')

    for _ in range(args.epochs):
        for batch in train_dataloader:
            # Clear gradients for this new batch
            optimizer.zero_grad()

            # Forward pass
            output = embedder(
                one=batch['one'].to(args.device),
                two=batch['two'].to(args.device)
            )

            # Compute loss
            loss = loss_function(
                output,
                batch['sims'].to(args.device)
            ) / (batch['one'].shape[0] * dataset.n_similarities)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Learning rate adjustment
            scheduler.step()

            # Keep track of progress
            progress_bar.update()

            # Save the training signals to tensor board
            learning_rate = scheduler.get_last_lr()[0]

            writer.add_scalar('train/lr', learning_rate, progress_bar.n)
            writer.add_scalar('train/loss', loss.item(), progress_bar.n)

            # Once in a while, evaluate the current embedder state on the eval split
            steps_until_evaluation -= 1

            if steps_until_evaluation == 0:
                eval_loss = evaluate(
                    embedder,
                    eval_dataloader,
                    args.device,
                    loss_function,
                    dataset.n_similarities,
                )

                if best_eval_loss is None or eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss

                    save_weights(args.weights_filename, embedder)

                steps_until_evaluation = args.eval_each

                writer.add_scalar('eval/loss', eval_loss, progress_bar.n)

            writer.flush()

    # If we never evaluated, still save the learned weights
    if n_steps < args.eval_each:
        save_weights(args.weights_filename, embedder)

    writer.close()
    progress_bar.close()


if __name__ == '__main__':
    main()
