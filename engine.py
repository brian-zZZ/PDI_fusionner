import os
import csv
import copy
import time
import datetime
import scipy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model import ProModelFactory
from data_handler import prepare_data
from optim import build_optim_sched, Flooding, adjust_learning_rate
from utils import TensorboardLogger


def train_epoch(model, dataloader, optimizer, loss_function, device, epoch, flooding=False, b=.9, args=None):
    model.train()
    tqdm_bar = tqdm(dataloader, desc='Epoch {} - Train'.format(epoch))
    iter_losses = [] # loss in each iteration within one epoch
    for step, batch in enumerate(tqdm_bar):
        y_val, tokenized_sent, (x_atom, x_bonds, x_atom_index, x_bond_index, x_mask), (amino_list, amino_degree_list, amino_mask) = batch
        inputs = x_atom.float(), x_bonds.float(), x_atom_index.long(), x_bond_index.long(), x_mask.float(), \
                 tokenized_sent.long(), amino_list.float(), amino_degree_list.long(), amino_mask.float() # change data type
        inputs = [t.to(device) for t in inputs] # move
        inputs = [t.reshape(-1, *t.shape[2:]) for t in inputs] # reshape: (batch_size, 1) -> batch_size (merge first two dims togerther)
        pred = model(*inputs)
        loss = loss_function(pred, y_val.to(device).float().view(-1, 1))
        if flooding:
            loss = Flooding(loss, b)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter_losses.append(loss.detach().cpu().item())
        tqdm_bar.set_postfix(loss=np.mean(iter_losses))


def eval_epoch(model, dataloader, device, epoch, set_name):
    model.eval()
    eval_MAE_list = []
    eval_MSE_list = []
    pred_list = []
    true_list = []
    for batch in tqdm(dataloader, desc='Epoch {} - Eval {}'.format(epoch, set_name), leave=False):
        y_val, tokenized_sent, (x_atom, x_bonds, x_atom_index, x_bond_index, x_mask), (amino_list, amino_degree_list, amino_mask) = batch
        y_val = y_val.to(device).float()
        inputs = x_atom.float(), x_bonds.float(), x_atom_index.long(), x_bond_index.long(), x_mask.float(), \
                 tokenized_sent.long(), amino_list.float(), amino_degree_list.long(), amino_mask.float()
        inputs = [t.reshape(-1, *t.shape[2:]).to(device) for t in inputs]
        with torch.no_grad():
            pred = model(*inputs)

        MAE = F.l1_loss(pred, y_val.view(-1, 1), reduction='none')
        MSE = F.mse_loss(pred, y_val.view(-1, 1), reduction='none')
        eval_MAE_list.extend(MAE.data.squeeze().cpu().numpy())
        eval_MSE_list.extend(MSE.data.squeeze().cpu().numpy())
        pred_list.extend(pred.data.squeeze().cpu().numpy())
        true_list.extend(y_val.data.squeeze().cpu().numpy())
    corr = scipy.stats.pearsonr(pred_list, true_list)[0]

    return np.array(eval_MAE_list).mean(), np.array(eval_MSE_list).mean(), corr


def predicted_value(model, dataloader, device, set_name):
    model.eval()
    pred_list = []
    true_list = []
    for batch in tqdm(dataloader, desc='ForwardPass prediction - {}'.format(set_name), leave=False):
        y_val, tokenized_sent, (x_atom, x_bonds, x_atom_index, x_bond_index, x_mask), (amino_list, amino_degree_list, amino_mask) = batch
        y_val = y_val.float()
        inputs = x_atom.float(), x_bonds.float(), x_atom_index.long(), x_bond_index.long(), x_mask.float(), \
                 tokenized_sent.long(), amino_list.float(), amino_degree_list.long(), amino_mask.float()
        inputs = [t.reshape(-1, *t.shape[2:]).to(device) for t in inputs]
        with torch.no_grad():
            pred = model(*inputs)

        pred_list.extend(pred.data.squeeze().cpu().numpy())
        true_list.extend(y_val.data.squeeze().cpu().numpy())
        set_list = [set_name] * len(pred_list)
    return pred_list, true_list, set_list


def runner(args):
    # Prepare datasets
    print('=== '*7 ,"Preparing datasets", ' ==='*7)
    dataloader_pack, num_atom_features, num_bond_features= prepare_data(args)
    train_loader, valid_loader, test_test_loader, test_casf2013_loader, test_astex_loader = dataloader_pack

    model = ProModelFactory(args.radius, args.T, num_atom_features, num_bond_features, args.fingerprint_dim,
                            args.p_dropout, args.pro_seq_dim, args.pro_gat_dim, args.model_type, 
                            args.fusion_n_layers, args.fusion_n_heads, args.fusion_dropout, args)
    if args.continue_ft: # continues finetuning
        # for fusion
        model_state_dict = model.state_dict()
        loaded_state_dict = torch.load(args.pre_ft_pth, map_location=args.device)
        matched_state_dict = {k: v for k, v in loaded_state_dict.items() if (k in model_state_dict) \
                                                                    & (v.numel()==model_state_dict[k].numel())}

        model.load_state_dict(matched_state_dict, strict=False)
    model = model.to(args.device)
    loss_function = nn.MSELoss()
    optimizer, scheduler = build_optim_sched(model.parameters(), args)
    if args.tb:
        tb_logger = TensorboardLogger(os.path.join(args.tb_log, args.model_log_base.split('/')[-1]))

    # for param_name, param_value in model.named_parameters():
    #     print(param_name, ":", param_value.size())
    print('Model ({}) parameters:'.format(args.model_type.upper()), sum(param.numel() for param in model.parameters()))

    print('=== '*7 ,"Finetuing model", ' ==='*7)
    ts = time.time()
    plot_RMSE, plot_R = [], []
    best_valid_RMSE, best_epoch = 1e9, 0 # for saving
    best_test_RMSE, best_test_epoch = 1e9, 0
    saturated_epochs = 0  # for early-stopping
    for epoch in range(args.epochs):
        epoch_ts = time.time()
        train_epoch(model, train_loader, optimizer, loss_function, args.device, epoch, args.flooding, args.b, args)
        train_MAE, train_MSE, train_R = eval_epoch(model, train_loader, args.device, epoch, "train")
        valid_MAE, valid_MSE, valid_R = eval_epoch(model, valid_loader, args.device, epoch, "valid")
        test_MAE, test_MSE, test_R = eval_epoch(model, test_test_loader, args.device, epoch, "core2016")
        test_casf2013_MAE, test_casf2013_MSE, test_casf2013_R = eval_epoch(model, test_casf2013_loader, args.device, epoch, "casf2013")
        test_astex_MAE, test_astex_MSE, test_astex_R = eval_epoch(model, test_astex_loader, args.device, epoch, "astex")

        if args.warmup:
            scheduler.step()
        if args.reduce & args.nonwarmup:
            scheduler.step(valid_MSE)
        epoch_te = time.time()
        for para_group in optimizer.param_groups:
            lr = para_group['lr']

        train_RMSE, valid_RMSE, test_RMSE, test_casf2013_RMSE, test_astex_RMSE = \
                        np.sqrt(train_MSE), np.sqrt(valid_MSE), np.sqrt(test_MSE), np.sqrt(test_casf2013_MSE), np.sqrt(test_astex_MSE)
        if valid_RMSE < best_valid_RMSE:
            best_valid_RMSE = valid_RMSE
            best_epoch = epoch
            best_model = copy.deepcopy(model)
        if test_RMSE < best_test_RMSE:
            best_test_RMSE = test_RMSE
            best_test_epoch = epoch
            best_test_model = copy.deepcopy(model)

        RMSE_group = [train_RMSE, valid_RMSE, test_RMSE, test_casf2013_RMSE, test_astex_RMSE]
        R_group =[train_R, valid_R, test_R, test_casf2013_R, test_astex_R]
        # Logging
        RMSE_log = 'epoch: {}, train_RMSE:{:.4f}, valid_RMSE:{:.4f}, test_RMSE(2016):{:.4f}, test_RMSE(casf2013):{:.4f}, test_RMSE(astex):{:.4f}'.format(
                        epoch, *RMSE_group)
        R_log =  '{} train_R:{:.4f}, valid_R:{:.4f}, test_R(2016):{:.4f}, test_R(casf2013):{:.4f}, test_R(astex):{:.4f}, lr: {}'.format(
                        len('epoch: {},'.format(epoch))*' ', *R_group, lr)
        each_epoch_time = "------------ The {} epoch spend {}m-{}s ------------".format(epoch, int((epoch_te-epoch_ts)/60), int((epoch_te-epoch_ts)%60))
        print(RMSE_log, '\n', R_log, '\n', each_epoch_time)
        # Write logs
        f = open(os.path.join(args.model_log_base, 'run-logs.log'), 'a')
        f.write(RMSE_log+'\n')
        f.write(R_log+'\n')
        f.write(each_epoch_time+'\n')
        
        plot_RMSE.append([epoch, *RMSE_group])
        plot_R.append([epoch, *R_group])

        if args.tb:
            group = ['train', 'valid', 'test', 'casf2013', 'astex']
            tb_logger.update_group({f'{group[i]}_RMSE': v for i, v in enumerate(RMSE_group)}, group_head="RMSE")
            tb_logger.update_group({f'{group[i]}_R': v for i, v in enumerate(R_group)}, group_head='R')
            tb_logger.update(lr=lr, head='lr')
            tb_logger.set_step()
            tb_logger.flush()
        
        # Early-stopping monitoring
        if epoch != 0:
            if valid_RMSE > last_valid_RMSE or last_valid_RMSE < valid_RMSE <= 0.005: 
                saturated_epochs = saturated_epochs+1
            else:
                saturated_epochs = 0
        if saturated_epochs >= args.patience+5:
            early_stop_log = "! Early-stopping at {} epochs".format(epoch)
            print(early_stop_log)
            f.write(early_stop_log+'\n')
            break
        last_valid_RMSE = valid_RMSE

    # Save training stats (RMSE, R)
    with open(os.path.join(args.model_log_base, "RMSE-R_stats.csv"), 'w', newline='', encoding='utf-8') as stats_f:
        fwriter = csv.writer(stats_f)
        fwriter.writerow(["epoch", "train_RMSE", "valid_RMSE", "test_RMSE(core2016)", "test_RMSE(casf2013)", "test_RMSE(astex)",
                                   "train_R", "valid_R", "test_R(core2016)", "test_R(casf2013)", "test_R(astex)"])
        for i in range(len(plot_RMSE)):
            fwriter.writerow(plot_RMSE[i] + plot_R[i][1:])
    # Save the best model (valid)
    torch.save(best_model.state_dict(), os.path.join(args.model_log_base,
                                        'best-model-validRMSE{:.4f}-epoch{}.pth'.format(best_valid_RMSE, best_epoch)))
    # Save the best model (test)
    torch.save(best_test_model.state_dict(), os.path.join(args.model_log_base,
                                        'best-[test]-model-testRMSE{:.4f}-epoch{}.pth'.format(best_test_RMSE, best_test_epoch)))
    end_log = "Finish {} epochs fine-tuning in {}. ".format(epoch, str(datetime.timedelta(seconds=time.time()-ts))) + \
              "The best results: valid-RMSE: {} at epoch: {}, ".format(best_valid_RMSE, best_epoch) + \
              "test-RMSE: {} at epoch: {}".format(best_test_RMSE, best_test_epoch)
    f.write(end_log)
    f.close()
    print(end_log)

    return np.array(plot_RMSE), np.array(plot_R), best_model, best_test_model, dataloader_pack
