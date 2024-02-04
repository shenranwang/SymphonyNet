import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

# assign directory
directory = 'logs'
 
# iterate over files in
# that directory
def plot_training():
    files = [f for f in sorted(os.listdir(directory)) if "train_instrument" in f]
    files = sorted(files, key=lambda f: int(f.split("_")[-1][:-4].replace("1e", "")))  # scaling order
    plot_training_granular(files)
    plot_training_granular(files, log_scaled=True)
    plot_training_overview(files)
    plot_training_overview(files, log_scaled=True)
    

def plot_training_granular(files, log_scaled=False):
    fig, axs = plt.subplots(7,2, figsize=(10,20))
    axs = axs.flatten()
    i = 0
    for filename in files:
        # plt.figure()
        data = pd.DataFrame.from_dict(collect_values_from_file(filename))
        data = data[data["log_type"] == "train_inner"]
        epoch = data.pop("epoch")
        x = [i for i in range(len(epoch))]
        log_type = data.pop("log_type")
        for k in data:
            axs[i].plot(x, data[k], label=k)

        axs[i].set_xlabel("parameter update")
        axs[i].set_ylabel("losses")
        if log_scaled:
            axs[i].set_yscale("log")
        axs[i].legend()
        axs[i].set_title(filename)
        i += 1
    plt.tight_layout(pad=1.50)
    if log_scaled:
        save_file_path = "/plots/train_losses_log.png"
    else:
        save_file_path = "/plots/train_losses.png"
    fig.savefig(directory + save_file_path) # filename.replace("out", "png")
    
    
def plot_training_overview(files, log_scaled=False):
    fig, axs = plt.subplots(13,2, figsize=(10,40))
    axs = axs.flatten()
    i = 0
    x = [47, 48, 49, 50, 51, 52, 53]
    for filename in files:
        # plt.figure()
        data = pd.DataFrame.from_dict(collect_values_from_file(filename))
        data.pop("epoch")
        for log_type in ["train", "valid"]:
            type_data = data[data["log_type"] == log_type]
            type_data.pop("log_type")
            for k in type_data:
                axs[i].plot(x, type_data[k], label=k)
            axs[i].set_xlabel("epoch")
            axs[i].set_ylabel("losses")
            if log_scaled:
                axs[i].set_yscale("log")
            axs[i].legend()
            axs[i].set_title(f"{log_type} - {filename}")   
            i += 1
    plt.tight_layout(pad=1.50)
    if log_scaled:
        save_file_path = "/plots/train_overview_log.png"
    else:
        save_file_path = "/plots/train_overview.png"
    fig.savefig(directory + save_file_path) # filename.replace("out", "png")

                    
def collect_values_from_file(filename):
    loss_categories = {
        'epoch': [],
        'log_type': [],
        'loss': [],
        'evt_loss': [],
        'dur_loss': [],
        'trk_loss': [],
        'ins_loss': [],
        'instr_loss': [],
    }
    i = 0
    epoch = 47
    f = open(os.path.join(directory, filename), "r")
    lines = f.readlines()
    for line in lines:
        l = line.split("|")
        if len(l) > 2 and "train_inner" in l[2]:
            # float(re.search('(?<=\=).*?(?=\,)', text).group(0))
            loss_categories["epoch"].append(epoch)
            loss_categories["log_type"].append(l[2].strip())
            loss_data = [text.split("=") for text in l[3].split(" ") if "loss" in text]
            for loss_cat, loss in loss_data:
                loss_categories[loss_cat].append(float(loss[:-1]))
        elif len(l) > 2 and (" train " == l[2] or "valid" in l[2]):
            loss_categories["epoch"].append(epoch)
            loss_categories["log_type"].append(l[2].strip())
            if l[2].strip() == "train":
                epoch += 1
            loss_data = [text.strip().split(" ") for text in l if "loss" in text]
            for loss_cat, loss in loss_data:
                if "best_loss" in loss_cat:
                    continue
                loss_categories[loss_cat].append(float(loss))
    return loss_categories
    
                
# Create a figure and an axes
def plot_evt_instr_matrix(train_dir):
    # plt.figure(figsize=(6, 6))
    # ax = plt.gca()

    # x = torch.load(matrix_path)
    # Display the matrix
    # cax = ax.imshow((torch.mean(x.clone().detach(), dim=0)).cpu().numpy(), aspect='auto')

    # Add color bar
    # plt.colorbar(cax)

    # Show the plot
    # plt.show()
    
    import matplotlib.animation as animation
    import os
    import glob
    
    ims = []
    fig = plt.figure()
    # fig, axs = plt.subplots(1,2, figsize=(10,20))
    ax = plt.axes()
    files = glob.glob(f"{os.getcwd()}/ckpt/{train_dir}/evt_instr_matrix*.pt")
    
    for i,f in enumerate(sorted(files)):
        x = torch.load(f, map_location=torch.device('cpu'))
        x = np.log(x.clone().detach().cpu().numpy() + 1e-8)
        ims.append(x)
        # fn = f.split("/")[-1]
        # ax.set_title(f"{float(i / 2)}h elapsed. {fn}")
        # plt.colorbar(im)
        # if i == 0:
        #     plt.imshow((x.clone().detach() / 2).cpu().numpy(), aspect='auto')  # show an initial one first
        # ims.append(im)
        
    im = ax.imshow(ims[0], aspect='auto')
    cbar = plt.colorbar(im)
    title = ax.set_title(f"{train_dir} - Epoch XX")
    # im1 = axs[1].imshow(ims[0], aspect='auto')
    # cbar1 = plt.colorbar(im,ax=axs[1])
    # title1 = axs[1].set_title(f"{train_dir} - Epoch XX")
        
    def init():
        im.set_data(ims[0])
        # im1.set_data(ims[0])
        title.set_text(f"{train_dir} - Epoch {47}")
        # title1.set_text(f"{train_dir} - Epoch {47}")
        
    def updatefig(i):
        im.set_data(ims[i])
        # im1.set_data(ims[i])
        title.set_text(f"{train_dir} - Epoch {i + 47}")
        # title1.set_text(f"{train_dir} - Epoch {i + 47}")

    ani = animation.FuncAnimation(fig, updatefig, init_func=init, blit=False, interval=1000, frames=6, repeat_delay=1000)
    
    ani.save(f"{os.getcwd()}/ckpt/{train_dir}/{train_dir}.gif")
    

if __name__ == "__main__":
    # plot_training()
    for scaling in ["0", "1e6", "1e5", "1e4", "1e7", "1e3", "1e8", "1e2", "1e9", "1e1", "1e10", "1e0", "1e11"]:
        plot_evt_instr_matrix(train_dir=f"train_instrument_loss_add_{scaling}_bs128")
    