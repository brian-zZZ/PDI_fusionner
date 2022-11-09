""" Integrate optimizers and optimization strategies all togerter.
    Visable API outside to import.
    By: Brian Zhang, UCAS, 2022.
"""

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from optim import Ranger, Adan, RAdam, Lookahead
from optim import WarmupCosineScheduler

def build_optim_sched(model_params, args):
    optimizer, scheduler = None, None
    if args.optim == 'adam':
        optimizer = optim.Adam(model_params, args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'ranger': # ~= RAdam + LookAHead
        optimizer = Ranger(model_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adan':
        optimizer = Adan(model_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.98, 0.92, 0.99), eps=1e-8)
    elif args.optim == 'radam':
        optimizer = RAdam(model_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    if args.lookahead: # Wrap the base optimizer with lookahead
        optimizer = Lookahead(optimizer, la_steps=args.k, la_alpha=0.8)
    
    if args.warmup:
        print("WarmupCosineScheduler activated")
        scheduler = WarmupCosineScheduler(optimizer, args.epochs, args.warmup_epochs, args.lr, args.min_lr)
    if args.reduce & ~args.warmup:
        print("ReduceLROnPlateau Scheduler activated")
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.patience, factor=args.factor, min_lr=args.min_lr)

    return optimizer, scheduler

def Flooding(loss, b=0.9):
    loss = (loss - b).abs() + b
    return loss
