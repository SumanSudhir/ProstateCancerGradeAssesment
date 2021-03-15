import os
import argparse

from BiGAN import *


def parse_args():
    parser = argparse.ArgumentParser(description='BiGAN')


    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--split', type=int, default=0, help='split to train from 0,1,2,3')
    parser.add_argument('--csv_path', type=str, default='../../data/karolinska.csv', help='Path to the csv file containing all image id')
    parser.add_argument('--files_path', type=str, default='../../CLAM/pandaPatches10x/patches', help='Directory containing all the patches')
    parser.add_argument('--save_dir', type=str, default='../OUTPUT/models', help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='../OUTPUT/results', help='Directory name to save the genrated images')
    parser.add_argument('--log_dir', type=str, default='../OUTPUT/logs', help='Directory to save training logs')

    # hyperparameters
    parser.add_argument('--epoch', type=int, default=25, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=16, help='The size of batch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='for adam')
    parser.add_argument('--slope', type=float, default=1e-2, help='for leaky ReLU')
    parser.add_argument('--decay', type=float, default=2.5*1e-5, help='for weight decay')
    parser.add_argument('--dropout', type=float, default=0.2)

    # network parameters
    parser.add_argument('--z_dim', type=int, default=128, help='The dimension of latent space Z')
    parser.add_argument('--h_dim', type=int, default=1024, help='The dimension of the hidden layers in case of a FC network')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


def main():
    # parse arguments
    args = parse_args()
    args.gpu = True
    if args is None:
        exit()

    bigan = BiGAN(args)

    # ecrase anciens fichiers
    with open('pixel_error_BIGAN.txt', 'w') as f:
        f.writelines('')
    with open('z_error_BIGAN.txt', 'w') as f:
        f.writelines('')

    bigan.train()
    print(" [*] Training finished!")

    bigan.save_model()


    # bigan.plot_states()


if __name__ == '__main__':
    main()
