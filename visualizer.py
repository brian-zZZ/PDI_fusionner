import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from sklearn.linear_model import LinearRegression
from rdkit.Chem import Draw
from rdkit import Chem

from engine import predicted_value

# set Times New Roman font globally
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def plot_curve(plot_RMSE, plot_R, model_type, dir_save):
    plot_RMSE, plot_R = plot_RMSE.T, plot_R.T
    plt.figure('1')
    plt.plot(plot_RMSE[0], plot_RMSE[1], label="train")
    plt.plot(plot_RMSE[0], plot_RMSE[2], label="valid")
    plt.plot(plot_RMSE[0], plot_RMSE[3], label="core2016")
    plt.plot(plot_RMSE[0], plot_RMSE[4], label="CASF2013")
    plt.plot(plot_RMSE[0], plot_RMSE[5], label="astex")
    plt.legend()
    plt.title("Datasets' RMSE of {} model".format(model_type))
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    plt.savefig(dir_save+'/Datasets-RMSE.jpg')
    plt.figure('2')
    plt.plot(plot_R[0], plot_R[1], label="train")
    plt.plot(plot_R[0], plot_R[2], label="valid")
    plt.plot(plot_R[0], plot_R[3], label="core2016")
    plt.plot(plot_R[0], plot_R[4], label="CASF2013")
    plt.plot(plot_R[0], plot_R[5], label="astex")
    plt.legend()
    plt.title("Datasets' R of {} model".format(model_type))
    plt.xlabel('epoch')
    plt.ylabel('R')
    plt.savefig(dir_save+'/Datasets-R.jpg')

    
def linreg_fitting(labels_df, dir_save):
    # Plot linear-regression fitting map
    # sns.set(color_codes=True)
    sns.set(context='paper', style='white')
    sns.set_color_codes() 
    # set_colors = {'train': 'b', 'valid': 'green', 'core2016': 'purple', 'casf2013': 'darkorange', 'astex': 'r'}
    set_colors = {'train': 'b', 'valid': 'r', 'core2016': 'b', 'casf2013': 'green', 'astex': 'orange'}
    for set_name, table in labels_df.groupby('set'):
        rmse = ((table['predicted'] - table['real']) ** 2).mean() ** 0.5
        mae = (np.abs(table['predicted'] - table['real'])).mean()
        corr = scipy.stats.pearsonr(table['predicted'], table['real'])
        lr = LinearRegression()
        lr.fit(table[['predicted']], table['real'])
        y_ = lr.predict(table[['predicted']])
        sd = (((table["real"] - y_) ** 2).sum() / (len(table) - 1)) ** 0.5
        print("%10s set: RMSE=%.3f, MAE=%.3f, R=%.3f (p=%.2e), SD=%.3f" %
            (set_name, rmse, mae, *corr, sd))
        
        grid = sns.jointplot(x='real', y='predicted', data=table, color=set_colors[set_name],
                            space=0, height=4, ratio=4, s=20, edgecolor='w', ylim=(0, 16), xlim=(0, 16))  # (0.16)
        grid.ax_joint.set_xticks(range(0, 16, 5))
        grid.ax_joint.set_yticks(range(0, 16, 5)) 
        grid.ax_joint.text(1, 14, set_name + ' set', fontsize=16)
        parm_font_size = 8
        grid.ax_joint.text(16, 19.5, 'RMSE: %.3f' % (rmse), fontsize=parm_font_size)
        grid.ax_joint.text(16, 18.5, 'MAE: %.3f' % (mae), fontsize=parm_font_size)
        grid.ax_joint.text(16, 17.5, 'R: %.3f ' % corr[0], fontsize=parm_font_size)
        grid.ax_joint.text(16, 16.5, 'SD: %.3f ' % sd, fontsize=parm_font_size)
        grid.ax_joint.text(16.5, -1.25, '$\it{(pK_a)}$')
        # grid.ax_joint.text(-2, 17, '(pKa)')
        grid.fig.savefig(dir_save+'/%s_linreg_fitting.jpg' % set_name, dpi=400)

def plot_hist(dir_save):
    def label_bars(ax, heights, rects, **kwargs):
        """Attach a text label on bottom of each bar."""
        for height, rect in zip(heights, rects):
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, -4),  # 4 points vertical offset.
                        textcoords='offset points',
                        ha='center', va='top', **kwargs)

    # RMSE, R, SD of train, valid, test, casf2013, astex
    pafnucy = np.array([[1.21, 0.77, 1.19],
                        [1.44, 0.72, 1.43],
                        [1.42, 0.78, 1.37],
                        [1.62, 0.70, 1.61],
                        [1.43, 0.57, 1.43]])
    ours = np.array([[0.967, 0.862, 0.928],
                     [1.403, 0.739, 1.350],
                     [1.290, 0.803, 1.287],
                     [1.506, 0.749, 1.500],
                     [1.373, 0.687, 1.262]])

    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(20, 10))
    x = np.arange(4)
    width = .28
    xticklabels = ['Valid', 'Core2016', 'CASF2013', 'Astex']
    spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1, 1], wspace=0.1, hspace=0.15)
    ### uppper bar, R
    ax1 = fig.add_subplot(spec[0])
    a1b1 = ax1.bar(x - width/2, pafnucy[1: , 1], width*0.8, label='Pafnucy', color='#3D57A4',
                    edgecolor='black', linewidth=2.5, ecolor='black', capsize=10)
    a1b2 = ax1.bar(x + width/2, ours[1: , 1], width*0.8, label='Ours', color='#EC1D25',
                    edgecolor='black', linewidth=2.5, ecolor='black', capsize=10)
    ax1.bar_label(a1b1, fmt='%.2f')
    ax1.bar_label(a1b2, fmt='%.3f', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(xticklabels, y=-.03)
    ax1.set_ylim([0.55, 0.85]) # [10, 80]
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax1.set_ylabel("R", fontsize=20)
    ax1.legend(fontsize=20)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['left'].set_linewidth(1.5)
    ### lower bar, RMSE
    ax2 = fig.add_subplot(spec[1])
    a2b1 = ax2.bar(x-width/2, pafnucy[1: , 0], width*0.8, color='#3D57A4',
                    edgecolor='black', linewidth=2.5, ecolor='black', capsize=10)
    a2b2 = ax2.bar(x+width/2, ours[1: , 0], width*0.8, color='#EC1D25',
                    edgecolor='black', linewidth=2.5, ecolor='black', capsize=10)
    
    label_bars(ax2, pafnucy[1:, 0], a2b1)
    label_bars(ax2, ours[1:, 0], a2b2, fontweight='bold')

    # ax2.bar_label(a2b2, fmt='%.3f')
    ax2.set_xticklabels([])
    ax2.xaxis.tick_top()
    ax2.set_ylim([1.25, 1.65])
    # ax2.set_yticks(np.linspace(0.75, 1, 6))
    ax2.invert_yaxis()
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_linewidth(1.5)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.set_ylabel("RMSE", fontsize=20)

    # plt.suptitle("Metrics of Ours vs Pafnucy", y=0.93, size=20)
    plt.savefig(dir_save, bbox_inches='tight', dpi=512)#, dpi=1024
    plt.show()
                


def plot_linreg(best_model, dataloader_pack, dir_save, device):
    # Forward pass to predict
    train_loader, valid_loader, test_test_loader, test_casf2013_loader, test_astex_loader = dataloader_pack
    pred_list_train, true_list_train, set_list_train = predicted_value(best_model, train_loader, device, "train")
    pred_list_valid, true_list_valid, set_list_valid = predicted_value(best_model, valid_loader, device, "valid")
    pred_list_core2016, true_list_core2016, set_list_core2016 = predicted_value(best_model, test_test_loader, device, "core2016")
    pred_list_casf2013, true_list_casf2013, set_list_casf2013 = predicted_value(best_model, test_casf2013_loader, device, "casf2013")
    pred_list_astex, true_list_astex, set_list_astex = predicted_value(best_model, test_astex_loader, device, "astex")
    all_list = {
                "predicted" : pred_list_train+pred_list_valid+pred_list_core2016+pred_list_casf2013+pred_list_astex, 
                "real" : true_list_train+true_list_valid+true_list_core2016+true_list_casf2013+true_list_astex,
                "set" : set_list_train+set_list_valid+set_list_core2016+set_list_casf2013+set_list_astex
            }
    labels_df = pd.DataFrame(all_list)
    labels_df.to_csv(dir_save + "/pred_real_labels.csv", header=True, index=False)

    linreg_fitting(labels_df, dir_save)

def plot_molecule(smis=None):
    if smis is None:
        smis=[
            'COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3',
            'C1=CC2=C(C(=C1)C3=CN=CN4C3=CC=C4)ON=C2C5=CC=C(C=C5)F',
            'COC(=O)C1=CC2=CC=CN2C=N1',
            'C1=C2C=C(N=CN2C(=C1)Cl)C(=O)O',
        ]
        # smis=[
        #     'C1=CC2=C(C(=C1)C3=CN=CN4C3=CC=C4)ON=C2C5=CC=C(C=C5)F',
        # ]
    mols=[]
    for i, smi in enumerate(smis):
        mol = Chem.MolFromSmiles(smi)
        mols.append(mol)
        img = Draw.MolsToGridImage(mols,molsPerRow=4,subImgSize=(200,200),legends=['' for x in mols])
        Draw.MolToFile(mol, "./files/mole_figs/molecule-{}.png".format(i))

    
if __name__ == '__main__':
    print("Plot linear-regression fitting of labels using the saved df")
    labels_df_pth = "./results_search/adan_nowarmup_reduce_pat10_fac0.5_minlr1e-5-bert_seed1145114_Sat-Nov--5-22:27:51-2022/pred_real_labels.csv"
    dir_save = "./results_search/adan_nowarmup_reduce_pat10_fac0.5_minlr1e-5-bert_seed1145114_Sat-Nov--5-22:27:51-2022/"
    labels_df = pd.read_csv(labels_df_pth)
    linreg_fitting(labels_df, dir_save)
    
    # plot_molecule()

    # plot_hist("./metric_comparison.jpg")
