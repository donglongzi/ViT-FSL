import numpy as np
import os
import glob
import argparse
import torch


def parse_args(script):
    parser = argparse.ArgumentParser(description='few-shot script %s' % (script))
    parser.add_argument('--dataset', default='CUB', help='CUB/miniImagenet/cifar/tieredImagenet/bit152')
    parser.add_argument('--model', default='resnet18', help='model:  WideResNet28_10/resnet18/resnet12/vit/vit_pre/bit_pre')
    parser.add_argument('--classifier', default='LR', help='LR/SVM')
    parser.add_argument('--train_aug', action='store_true',  help='perform data augmentation or not during training ')

    if script == 'train':
        parser.add_argument('--num_classes', default=100, type=int, help='total number of classes')
        parser.add_argument('--save_freq', default=10, type=int, help='Save frequency')
        parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch')
        parser.add_argument('--stop_epoch', default=100, type=int, help='Stopping epoch')
        parser.add_argument('--resume', action='store_true', help='continue from previous trained model with largest epoch')
        parser.add_argument('--lr', default=0.05, type=int, help='learning rate')
        parser.add_argument('--batch_size', default=64, type=int, help='batch size ')
        parser.add_argument('--test_batch_size', default=64, type=int, help='batch size ')
        parser.add_argument('--lr_decay_epochs', type=str, default='50,70,90', help='where to decay lr, can be a list')

    elif script == 'test':
        parser.add_argument('--lr', default=0.05, type=int, help='learning rate')
        parser.add_argument('--lr_decay_epochs', type=str, default='30,40', help='where to decay lr, can be a list')
        parser.add_argument('--num_classes', default=200, type=int, help='total number of classes')
        parser.add_argument('--n_test_runs', type=int, default=600, metavar='N', help='Number of test runs')
        parser.add_argument('--n_ways', type=int, default=15, metavar='N', help='Number of classes for doing each classification run')
        parser.add_argument('--n_shots', type=int, default=1, metavar='N', help='Number of shots in test')
        parser.add_argument('--n_queries', type=int, default=15, metavar='N', help='Number of query in test')
        parser.add_argument('--n_unlabelled', type=int, default=0, metavar='N', help='Number of unlabelled in test')
        parser.add_argument('--n_aug_support_samples', default=0, type=int, help='The number of augmented samples for each meta test sample')
        parser.add_argument('--num_workers', type=int, default=1, metavar='N', help='Number of workers for dataloader')
        parser.add_argument('--batch_size', default=64, type=int, help='batch size ')

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return args


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)

