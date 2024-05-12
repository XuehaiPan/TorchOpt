# Copyright 2022-2024 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This file is modified from:
# https://github.com/facebookresearch/higher/blob/main/examples/maml-omniglot.py
# ==============================================================================
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This example shows how to use TorchOpt to do Model Agnostic Meta Learning (MAML)
for few-shot Omniglot classification.
For more details see the original MAML paper:
https://arxiv.org/abs/1703.03400
This code has been modified from Jackie Loong's PyTorch MAML implementation:
https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot_train.py
Our MAML++ fork and experiments are available at:
https://github.com/bamos/HowToTrainYourMAMLPytorch
"""

import argparse
import functools
import pathlib
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchopt


CWD = pathlib.Path(__file__).absolute().parent
sys.path.append(str(CWD.parent / 'few-shot'))

from helpers.omniglot_loaders import OmniglotNShot


mpl.use('Agg')
plt.style.use('bmh')


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument(
        '--task_num',
        type=int,
        help='meta batch size, namely task num',
        default=32,
    )
    argparser.add_argument('--seed', type=int, help='random seed', default=1)
    args = argparser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Set up the Omniglot loader.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    db = OmniglotNShot(
        '/tmp/omniglot-data',
        batchsz=args.task_num,
        n_way=args.n_way,
        k_shot=args.k_spt,
        k_query=args.k_qry,
        imgsz=28,
        rng=rng,
        device=device,
    )

    # Create a vanilla PyTorch neural network.
    inplace_relu = True
    net = nn.Sequential(
        nn.Conv2d(1, 64, 3),
        nn.BatchNorm2d(64, affine=True, track_running_stats=False),
        nn.ReLU(inplace=inplace_relu),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, 3),
        nn.BatchNorm2d(64, affine=True, track_running_stats=False),
        nn.ReLU(inplace=inplace_relu),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, 3),
        nn.BatchNorm2d(64, affine=True, track_running_stats=False),
        nn.ReLU(inplace=inplace_relu),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64, args.n_way),
    ).to(device)

    net.train()
    params = dict(net.named_parameters())
    buffers = dict(net.named_buffers())
    net.to('meta')

    # We will use Adam to (meta-)optimize the initial parameters
    # to be adapted.
    meta_opt = optim.Adam(params.values(), lr=1e-3)

    log = []
    test(db, (net, params, buffers), epoch=-1, log=log)
    for epoch in range(10):
        train(db, (net, params, buffers), meta_opt, epoch=epoch, log=log)
        test(db, (net, params, buffers), epoch=epoch, log=log)
        plot(log)


# Trains a model for n_inner_iter using the support and returns a loss
# using the query.
def loss_for_task(net, n_inner_iter, x_spt, y_spt, x_qry, y_qry):
    net, params, buffers = net
    opt = torchopt.sgd(lr=1e-1)
    opt_state = opt.init(params)

    def compute_loss(new_params, buffers, x, y):
        logits = torch.func.functional_call(net, (new_params, buffers), x)
        loss = F.cross_entropy(logits, y)
        return loss

    new_params = params
    # change sgd implementation with torchopt
    for _ in range(n_inner_iter):
        grads = torch.func.grad(compute_loss)(new_params, buffers, x_spt, y_spt)
        updates, opt_state = opt.update(grads, opt_state, inplace=False)
        new_params = torchopt.apply_updates(new_params, updates, inplace=False)

    # The final set of adapted parameters will induce some
    # final loss and accuracy on the query dataset.
    # These will be used to update the model's meta-parameters.
    qry_logits = torch.func.functional_call(net, (new_params, buffers), x_qry)
    qry_loss = F.cross_entropy(qry_logits, y_qry)
    qry_acc = (qry_logits.argmax(dim=1) == y_qry).float().mean()

    return qry_loss, qry_acc


def train(db, net, meta_opt, epoch, log):
    net, params, buffers = net
    n_train_iter = db.x_train.shape[0] // db.batchsz

    for batch_idx in range(n_train_iter):
        start_time = time.time()
        # Sample a batch of support and query images and labels.
        x_spt, y_spt, x_qry, y_qry = db.next()

        n_inner_iter = 5
        meta_opt.zero_grad()

        # In parallel, trains one model per task. There is a support (x, y)
        # for each task and a query (x, y) for each task.
        compute_loss_for_task = functools.partial(
            loss_for_task, (net, params, buffers), n_inner_iter
        )
        qry_losses, qry_accs = torch.func.vmap(compute_loss_for_task)(x_spt, y_spt, x_qry, y_qry)

        # Compute the maml loss by summing together the returned losses.
        qry_losses = torch.mean(qry_losses)
        qry_losses.backward()
        meta_opt.step()
        qry_losses = qry_losses.item()
        qry_accs = 100.0 * torch.mean(qry_accs).item()
        i = epoch + float(batch_idx) / n_train_iter
        iter_time = time.time() - start_time

        if batch_idx % 4 == 0:
            print(
                f'[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}',
            )
        log.append(
            {
                'epoch': i,
                'loss': qry_losses,
                'acc': qry_accs,
                'mode': 'train',
                'time': time.time(),
            },
        )


def test(db, net, epoch, log):
    # Crucially in our testing procedure here, we do *not* fine-tune
    # the model during testing for simplicity.
    # Most research papers using MAML for this task do an extra
    # stage of fine-tuning here that should be added if you are
    # adapting this code for research.
    net, params, buffers = net
    n_test_iter = db.x_test.shape[0] // db.batchsz

    qry_losses = []
    qry_accs = []

    for _ in range(n_test_iter):
        x_spt, y_spt, x_qry, y_qry = db.next('test')
        task_num, setsz, c_, h, w = x_spt.size()

        # TODO: Maybe pull this out into a separate module so it
        # doesn't have to be duplicated between `train` and `test`?
        n_inner_iter = 5

        for i in range(task_num):
            new_params = params
            for _ in range(n_inner_iter):
                spt_logits = torch.func.functional_call(net, (new_params, buffers), x_spt[i])
                spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                grads = torch.autograd.grad(spt_loss, tuple(new_params.values()))
                new_params = {
                    name: p - 1e-1 * g for (name, p), g, in zip(new_params.items(), grads)
                }

            # The query loss and acc induced by these parameters.
            qry_logits = torch.func.functional_call(net, (new_params, buffers), x_qry[i])
            qry_loss = F.cross_entropy(qry_logits, y_qry[i], reduction='none')
            qry_losses.append(qry_loss.detach())
            qry_accs.append((qry_logits.argmax(dim=1) == y_qry[i]).detach())

    qry_losses = torch.mean(torch.stack(qry_losses)).item()
    qry_accs = 100.0 * torch.mean(torch.stack(qry_accs).float()).item()

    print(f'[Epoch {epoch+1:.2f}] Test Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}')
    log.append(
        {
            'epoch': epoch + 1,
            'loss': qry_losses,
            'acc': qry_accs,
            'mode': 'test',
            'time': time.time(),
        },
    )


def plot(log):
    # Generally you should pull your plotting code out of your training
    # script but we are doing it here for brevity.
    df = pd.DataFrame(log)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=250)
    train_df = df[df['mode'] == 'train']
    test_df = df[df['mode'] == 'test']
    ax.plot(train_df['epoch'], train_df['acc'], label='Train')
    ax.plot(test_df['epoch'], test_df['acc'], label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(85, 100)
    ax.set_title('MAML Omniglot')
    ax.legend(ncol=2, loc='lower right')
    fig.tight_layout()
    fname = 'maml-accs.png'
    print(f'--- Plotting accuracy to {fname}')
    fig.savefig(fname)
    plt.close(fig)


if __name__ == '__main__':
    main()
