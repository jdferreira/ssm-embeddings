from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Graph the loss values (for training and evaluation) of an experiment.'
    )

    parser.add_argument(
        'dirname',
        help='The directory where the outputs of training is stored'
    )

    parser.add_argument(
        '-o', '--output', default='loss.png',
        help='The filename to store the graph, relative to the directory given as argument. '
             'Defaults to "loss.png"'
    )

    parser.add_argument(
        '-l', '--log', action='store_true',
        help='Use a log scale on the y axis.'
    )

    parser.add_argument(
        '-m', '--max', type=float,
        help='The maximum y value to plot. If unspecified, the curve shows all values.'
    )

    args = parser.parse_args()

    args.output = os.path.join(args.dirname, args.output)

    return args


def get_steps_per_epoch(args):
    with open(os.path.join(args.dirname, 'config.json')) as f:
        return json.load(f)['steps_per_epoch']


def main():
    sns.set_theme()

    args = get_arguments()

    steps_per_epoch = get_steps_per_epoch(args)

    ea = event_accumulator.EventAccumulator(
        args.dirname,
        size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        }
    )

    ea.Reload()

    train_loss = pd.DataFrame(
        ea.Scalars('train/loss')
    ).drop(columns=['wall_time'])

    eval_loss = pd.DataFrame(
        ea.Scalars('eval/loss')
    ).drop(columns=['wall_time'])

    df = train_loss.merge(
        eval_loss,
        on='step',
        how='outer',
        suffixes=('_train', '_eval')
    )

    df.columns = ['Step', 'Train', 'Eval']
    df['Epoch'] = df['Step'] / steps_per_epoch

    plot = sns.lineplot(
        data=df.melt(
            var_name='Type',
            value_name='Loss',
            id_vars=['Step', 'Epoch']
        ),
        x='Epoch',
        y='Loss',
        hue='Type',
        markers=True,
    )

    if args.log:
        plot.set(yscale='log')

        if args.max:
            plot.set(ylim=(None, args.max))

    else:
        plot.set(ylim=(0, args.max))

    plot.get_figure().savefig(args.output)


if __name__ == '__main__':
    main()
