import yaml
import time
import os
import torch
from argparse import ArgumentParser

from engine import runner
from utils import set_seed
from visualizer import plot_curve, plot_linreg

def get_args():
    """ Parameters configuratation """
    parser = ArgumentParser(description='Model configuration')
    # model params
    parser.add_argument("--model_idx", type=int, default=0, help='Index of model in model_list')
    parser.add_argument("--model_type", default=None, help="The type of model, will overwrite model_list[model_idx] if it's not None")
    parser.add_argument("--fusion_n_layers", type=int, default=1)
    parser.add_argument("--fusion_n_heads", type=int, default=4)
    parser.add_argument("--fusion_dropout", type=float, default=0.1)
    # optimization params
    parser.add_argument("--optim", type=str, default='adan', choices=['adam', 'ranger', 'adan', 'radam', 'adamw'])
    parser.add_argument("--lookahead", action='store_true', default=False, help='Whether or not use LookAHead for optimzation')
    parser.add_argument("--warmup", action='store_true', default=False, help='Warmup and then decrease lr in cosine function if `True`')
    parser.add_argument("--nonreduce", action='store_true', default=False, help='Do not use ReduceLROnPlateau if `True`')
    parser.add_argument("--epochs", type=int, default=150) # 
    parser.add_argument("--lr", type=float, default=1e-3, help='learning rate')
    parser.add_argument("--min_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--warmup_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20, help='Epochs to wait for lr-reducing and ealry-stopping')
    parser.add_argument("--nonflooding", action='store_true', default=False, help='Do not apply flooding optimization strategy if `True`')
    parser.add_argument("--b", type=float, default=.9, help='Coefficient factor b of flooding')
    parser.add_argument("--k", type=int, default=5, help='Steps to look in LookAHead, default: 5')
    parser.add_argument("--factor", type=float, default=0.1, help='Factor for lr reducer')
    # other params
    parser.add_argument("--gpu_start", type=int, default=0, help='Id of gpu')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1145114, help='Random seed for reproducing')
    parser.add_argument("--continue_ft", action='store_true', default=False, help='Whether or not contiunes FT from pre-FT weights')
    parser.add_argument("--prefix", type=str, default='', help='prefix of result log')
    parser.add_argument("--sampling", action='store_true', help='Whether or not sampling')
    parser.add_argument("--tb", action='store_true', default=False, help='Activate TensorBoard to log and plot')
    args = parser.parse_args()

    # Load config from yaml file
    config = yaml.load(open("./config.yaml", 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    for k, v in config.items():
        setattr(args, k, v)

    args.device = torch.device("cuda:{}".format(args.gpu_start) if torch.cuda.is_available() else "cpu")
    if args.model_type is None:
        args.model_type = args.model_list[args.model_idx]
    
    # boolean params conversion for convinience
    args.reduce = not(args.nonreduce)
    args.flooding = not(args.nonflooding)

    current_time = str(time.ctime()).replace(' ', '-')
    args.model_log_base = os.path.join(args.log_base, '{}-{}_seed{}_'.format(args.prefix, args.model_type, args.seed)+current_time)
    os.makedirs(args.model_log_base, exist_ok=True)
    
    return args

def main(args):
    set_seed(args.seed)
    # Fine-tuning process
    print("Start fine-tuing procedure")
    plot_RMSE, plot_R, best_model, best_test_model, dataloader_pack = runner(args)
    
    print('=== '*7 ,"Plotting figures", ' ==='*7)
    # Plot curves of training stats
    plot_curve(plot_RMSE, plot_R, args.model_type, args.model_log_base)
    # Plot linear-regression fitting map
    print("Single forward-pass to predict labels then linear-regression fitting with the true labels")
    # plot_linreg(best_model, dataloader_pack, args.model_log_base, args.device) # best valid
    plot_linreg(best_test_model, dataloader_pack, args.model_log_base, args.device) # best test

if __name__ == '__main__':
    args = get_args()
    main(args)
