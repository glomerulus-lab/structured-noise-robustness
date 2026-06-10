import os
import pandas as pd 
import pickle 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import glob
import torch
import re

figures_folder = "../saved/_figures/"

# Takes a list of tickmarks and returns a list of string labels where every other one is the empty string to avoid crowding
def ticks_to_labels(ticks):
    result = [""] * len(ticks) # Initialize list
    for i in range(len(ticks)):
        if i % 2 != 0:
            result[i] = "%.2f" % ticks[i]
    return result

def make_it_works_table(df, filename):
    """
    Creates a table where different noise conditions are columns
    and each row corresponds to a unique experimental setup.
    """
    # Prepare Dataframe
    group_columns = ['Data Alteration', 'trace', 'attack', "Intensity", "noise"]
    df = df[(df["covadv"] == df["attack"]) | (df["attack"] == "transformed")]
    df['Data Alteration'] =  df['covtrans'].combine_first(df['covadv'])
    df.reset_index(inplace=True)
    convert_dict = {"covcorr_sev": float}
    df = df.astype(convert_dict)
    df['covtrans_scale'] = df['covtrans_scale'].replace(0, None)
    df['Intensity'] = df['covtrans_scale'].combine_first(df['covcorr_sev']).combine_first(df['eps'])
    df.drop(columns=["covtrans_scale", "eps", "covcorr_sev", "covtrans", "covadv", "rotate", "transform_scale", "arch", "n_ensemble", "noisy_layer", "alpha", "beta", "ac_eps", "covrot", "dataset", "corr_sev", "transform"], inplace=True)
    df.reset_index(inplace=True)

    ci_df = df.groupby(group_columns)["accuracy"].describe()[["count", "mean", "std"]].reset_index()
    ci_df["lower_ci"] = ci_df["mean"] - 1.96*(ci_df["std"]/np.sqrt(ci_df["count"]))
    ci_df["upper_ci"] = ci_df["mean"] + 1.96*(ci_df["std"]/np.sqrt(ci_df["count"]))
    ci_df["Result"] = ci_df["mean"].round(2).astype(str) + " CI=[" + ci_df["lower_ci"].round(2).astype(str) + "," + ci_df["upper_ci"].round(2).astype(str) + "]"
    ci_df.drop(columns=["lower_ci", "upper_ci", "mean", "count", "std"], inplace=True)
    
    # Check for duplicate columns
    dup_counts = (
        ci_df
        .groupby(group_columns)
        .size()
        .reset_index(name='n')
    )

    # Any groups with more than one result?
    duplicates = dup_counts[dup_counts['n'] > 1]
    if len(duplicates > 0):
        print("You've got duplicate columns and this is going to break pivot. ")
        exit()

    ci_df_wide = (
        ci_df
        .pivot(index=['Data Alteration', 'trace', 'attack', 'Intensity'], columns='noise', values='Result')
        .reset_index()
    )

    # Rename columns to match desired titles
    ci_df_wide = ci_df_wide.rename(columns={
        "Full Cov": "Full Cov Result",
        "Diagonal": "Diagonal Result",
        "Identity": "Identity Result",
        "No Noise": "No Noise Result"
    })
    # Save to Excel
    ci_df_wide.to_excel(f"{figures_folder}{filename}.xlsx", index=False, header=True)

def mismatch_mod_hm(df, filename):
    df = df[df["noise"] == "Full Cov"]
    df = df[df["attack"] == "transformed"]
    df.drop(columns=["attack", "noisy_layer", "n_ensemble", "trace", "corr_sev", "covcorr_sev", "noise", "arch", "eps", "ac_eps", "covrot", "covadv", "alpha", "beta"], inplace=True)
    tmp = df[df["transform"] == "brightness"]
    print(tmp[tmp["covtrans"] == "gaussian_noise"])
    mean_df = df.groupby(["covtrans", "transform"])["accuracy"].describe()[["mean"]].reset_index()
    
    transform_names = ['brightness', 'contrast', 'elastic', 'gaussian_noise', 'impulse_noise', 'motion_blur', 'obstruction', 'perspective', 'rotate',  'snow']
    transform_titles = ['Brightness', 'Contrast', 'Elastic', 'Gaussian Noise', 'Impulse Noise', 'Motion Blur', 'Obstruction', 'Perspective', 'Rotate',  'Snow']

    # One column for each covariance, one row for each data alteration 
    hm_df = pd.DataFrame(columns=["covtrans"] + transform_names)
    for transform in transform_names:
        row_df = mean_df[mean_df["transform"] == transform]
        assert(len(row_df) == len(transform_names))
        row = [transform] + [row_df[row_df["covtrans"] == covtrans]["mean"].item() for covtrans in transform_names]
        hm_df.loc[len(hm_df)] = row
    hm_df = hm_df.set_index("covtrans")

    fig, ax = plt.subplots(figsize=(14, 10))
    s = sns.heatmap(hm_df, annot=True, cmap='YlGnBu', square=True)
    
    colorbar = ax.collections[0].colorbar
    colorbar.set_label("Test Accuracy", fontsize=11)  
    
    ax.set_xlabel("Covariance Source", fontsize=13)
    ax.set_ylabel("Applied Image Modification", fontsize=13)

    ax.set_xticklabels(transform_titles)
    ax.set_yticklabels(transform_titles)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(f"{figures_folder}{filename}.png", dpi=150.)

def it_works_plot(title, df, error_bars=None):
    df = df[df['dataset'] != 'train']
    df["label"] = df["noise"]
    plt.tight_layout()
    sns.set_style("whitegrid")
    #define dimensions of subplots (rows, columns)
    fig, ax = plt.subplots(1, 3, figsize=(13, 5))
    # create labels:
    labels = ["Full Cov","Diagonal", "Identity", "No Noise"]

    n_colors = 1
    #create color palette
    palette = []
    palette.extend(sns.color_palette(palette="Reds", n_colors=n_colors+1).as_hex()[1:])    
    palette.extend(sns.color_palette(palette="Greens", n_colors=n_colors+1).as_hex()[1:])
    palette.extend(sns.color_palette(palette="Purples", n_colors=n_colors+1).as_hex()[1:])   
    palette.append("#000000")

    for i in range(3):
        if i == 0: # 
            subplot_title = "AutoPGD Attack, tr=2.0"
            attack = "AutoPGD"
            x = "eps"
            x_ticks = [0.001, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
            x_tick_labels = [0.001, "", 0.04, "", 0.08, "", 0.12, "", 0.16, "", 0.2]
            x_tick_rot = 0
            x_label = "Attack Strength $\epsilon$"

            plot_data = df[df['attack'] == attack]
            plot_data = plot_data[plot_data['trace'] > 1.6]
            plot_data = plot_data[plot_data['covadv'] == attack]
        elif i == 1: 
            subplot_title = "Motion Blur, tr=0.5"
            corruption = "motion_blur"
            x = "corr_sev"
            x_ticks = [1, 2, 3, 4, 5]
            x_tick_labels = x_ticks
            x_tick_rot = 0
            x_label = "Severity"
            plot_data = df[df['attack'] == "transformed"]
            plot_data = plot_data[plot_data['covtrans'] == corruption]
            plot_data = plot_data[plot_data['trace'] < 1.0]


        elif i == 2: 
            subplot_title = "Obstruction, tr=0.5"
            transform = "obstruction"
            x = "covtrans_scale"
            x_ticks = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
            x_tick_labels = ["", 0.4, "", 0.8,"", 1.2, "", 1.6,"", 2.0]
            x_tick_rot = 0
            x_label = "Scale"
            plot_data = df[df['attack'] == "transformed"]
            plot_data = plot_data[plot_data['covtrans'] == transform]
            plot_data = plot_data[plot_data['trace'] < 1.0]


        s = sns.lineplot(
            ax=ax[i],
            data=plot_data, x=x, y="accuracy", errorbar="ci",
            hue="label", legend=(False if i != 0 else "full"),
            hue_order=labels,
            palette=palette,
        )

        # Use dashed line for no noise         
        lines = ax[i].get_lines()
        lines[3].set_linestyle('--')

        # Change the legend entry to match
        legend = ax[i].get_legend()
        if legend is not None:
            for legline in legend.get_lines():
                if legline.get_label() == "No Noise":
                    legline.set_linestyle("--")

        if i== 0:
            sns.move_legend(ax[i], "upper left", bbox_to_anchor=(3.47, 1), title="Noise Cov")
        ax[i].set_title(subplot_title)

        # Set y
        y_ticks = np.arange(start=0, stop=0.95, step=0.05)
        ax[i].set_yticks(y_ticks)
        ax[i].set_ylim(bottom=0, top=0.925)
 
        if i == 0:
            y_tick_labels = y_ticks.tolist()
            for j in range(len(y_tick_labels)):

                if j % 2 == 0:
                    y_tick_labels[j] = ""
                else:
                    y_tick_labels[j] = "%.2f" % y_ticks[j]

            ax[i].set_ylabel("Test Accuracy")
            ax[i].set_yticklabels(y_tick_labels)
        else: 
            ax[i].set_ylabel("")
            ax[i].set_yticklabels([""] * len(y_ticks))

        # Set x
        ax[i].set_xticks(x_ticks) 
        ax[i].set_xlim(left=x_ticks[0], right=x_ticks[-1])
        s.set_xticklabels(rotation=x_tick_rot, labels=x_tick_labels) 
        ax[i].set_xlabel(x_label)

    fig.supxlabel("Modification Strength", fontsize="large", y=0.02, x=0.46)
    fig.subplots_adjust(right=0.8, bottom=0.157)
    full_title = f'{title}'
    plt.savefig(f"{figures_folder}{full_title.replace(":", "").replace(",", "").replace(' ','_')}.png")

def adv_layers_plot(title, df):
    df = df[df['dataset'] != 'train']
    df = df[df['attack'] == "benign"]
    df = df[df['covadv'] == "AutoPGD"]
    df["label"] = df["noise"] + ", L=" + df["noisy_layer"].astype(str)
    df = df.replace(to_replace="No Noise, L=0", value="No Noise")
    df = df.replace(to_replace="No Noise, L=1", value="No Noise")
    df = df.replace(to_replace="No Noise, L=2", value="No Noise")
    df = df.replace(to_replace="No Noise, L=3", value="No Noise")
    df = df.replace(to_replace="No Noise, L=4", value="No Noise")

    # print(sub)
    plt.tight_layout()
    sns.set_style("whitegrid")

    #define dimensions of subplots (rows, columns)
    fig, ax = plt.subplots(figsize=(9, 9))

    # create labels:
    noise_settings = ["Identity", "Diagonal", "Full Cov"]
    layer_settings = ["0", "1", "2", "3", "4"]
    labels = ["No Noise"]
    for layer in layer_settings:
            for noise in noise_settings:
                labels.append(f"{noise}, L={layer}")
    n_colors = 3
    #create color palette
    palette = ["#000000"]
    palette.extend(sns.color_palette(palette="Reds", n_colors=n_colors+1).as_hex()[1:])
    palette.extend(sns.color_palette(palette="Greens", n_colors=n_colors+1).as_hex()[1:])
    palette.extend(sns.color_palette(palette="Blues", n_colors=n_colors+1).as_hex()[1:])        
    palette.extend(sns.color_palette(palette="Purples", n_colors=n_colors+1).as_hex()[1:])
    palette.extend(sns.color_palette(palette="Wistia", n_colors=n_colors).as_hex())

    s = sns.lineplot(
        ax=ax,
        data=df, x="eps", y="accuracy",
        errorbar=None,
        hue="label", legend="full",
        hue_order=labels,
        palette=palette,
    )
    
    # Set x
    x_ticks = [0.001, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
    x_label = "Attack Strength $\epsilon$"
    ax.set_xticks(x_ticks) 
    ax.set_xlim(x_ticks[0], x_ticks[-1]) 
    s.set_xticklabels(x_ticks) 
    ax.set_xlabel(x_label)

    # Set y
    y_ticks = np.arange(start=0, stop=0.95, step=0.05)
    ax.set_yticks(y_ticks)
    ax.set_ylim(bottom=0, top=0.925)
    y_tick_labels = y_ticks.tolist()
    for j in range(len(y_tick_labels)):

        if j % 2 == 0:
            y_tick_labels[j] = ""
        else:
            y_tick_labels[j] = "%.2f" % y_ticks[j]

    ax.set_ylabel("Test Accuracy")
    ax.set_yticklabels(y_tick_labels)

    fig.subplots_adjust(right=0.85)
    plt.savefig(f"{figures_folder}{title.replace(":", "").replace(",", "").replace(' ','_')}.png")

def corruption_layers_plot(title, df):
    df = df[df['dataset'] != 'train']
    df = df[df['attack'] == "benign"]
    df = df[df['covtrans'] == "motion_blur"]
    df["label"] = df["noise"] + ", L=" + df["noisy_layer"].astype(str)
    df = df.replace(to_replace="No Noise, L=0", value="No Noise")
    df = df.replace(to_replace="No Noise, L=1", value="No Noise")
    df = df.replace(to_replace="No Noise, L=2", value="No Noise")
    df = df.replace(to_replace="No Noise, L=3", value="No Noise")
    df = df.replace(to_replace="No Noise, L=4", value="No Noise")

    # print(sub)
    plt.tight_layout()
    sns.set_style("whitegrid")

    #define dimensions of subplots (rows, columns)
    fig, ax = plt.subplots(figsize=(9, 9))

    # create labels:
    noise_settings = ["Identity", "Diagonal", "Full Cov"]
    layer_settings = ["0", "1", "2", "3", "4"]
    labels = ["No Noise"]
    for layer in layer_settings:
            for noise in noise_settings:
                labels.append(f"{noise}, L={layer}")
    n_colors = 3

    #create color palette
    palette = ["#000000"]
    palette.extend(sns.color_palette(palette="Reds", n_colors=n_colors+1).as_hex()[1:])
    palette.extend(sns.color_palette(palette="Greens", n_colors=n_colors+1).as_hex()[1:])
    palette.extend(sns.color_palette(palette="Blues", n_colors=n_colors+1).as_hex()[1:])        
    palette.extend(sns.color_palette(palette="Purples", n_colors=n_colors+1).as_hex()[1:])
    palette.extend(sns.color_palette(palette="Wistia", n_colors=n_colors).as_hex())

    s = sns.lineplot(
        ax=ax,
        data=df, x="covcorr_sev", y="accuracy",
        errorbar=None,
        hue="label", legend="full",
        hue_order=labels,
        palette=palette,
    )
    
    # Set x
    x_ticks = [1, 2, 3, 4, 5]
    x_label = "Severity"
    ax.set_xticks(x_ticks) 
    ax.set_xlim(x_ticks[0], x_ticks[-1]) 
    s.set_xticklabels(x_ticks) 
    ax.set_xlabel(x_label)

    # Set y
    y_ticks = np.arange(start=0, stop=0.95, step=0.05)
    ax.set_yticks(y_ticks)
    ax.set_ylim(bottom=0, top=0.925)
    y_tick_labels = y_ticks.tolist()
    for j in range(len(y_tick_labels)):

        if j % 2 == 0:
            y_tick_labels[j] = ""
        else:
            y_tick_labels[j] = "%.2f" % y_ticks[j]

    ax.set_ylabel("Test Accuracy")
    ax.set_yticklabels(y_tick_labels)

    fig.subplots_adjust(right=0.85)
    plt.savefig(f"{figures_folder}{title.replace(":", "").replace(",", "").replace(' ','_')}.png")

def obstruction_layers_plot(title, df):
    df = df[df['dataset'] != 'train']
    df = df[df['attack'] == "benign"]
    df = df[df['covtrans'] == "obstruction"]
    df["label"] = df["noise"] + ", L=" + df["noisy_layer"].astype(str)
    df = df.replace(to_replace="No Noise, L=0", value="No Noise")
    df = df.replace(to_replace="No Noise, L=1", value="No Noise")
    df = df.replace(to_replace="No Noise, L=2", value="No Noise")
    df = df.replace(to_replace="No Noise, L=3", value="No Noise")
    df = df.replace(to_replace="No Noise, L=4", value="No Noise")

    # print(sub)
    plt.tight_layout()
    sns.set_style("whitegrid")

    #define dimensions of subplots (rows, columns)
    fig, ax = plt.subplots(figsize=(9, 9))

    # create labels:
    noise_settings = ["Identity", "Diagonal", "Full Cov"]
    layer_settings = ["0", "1", "2", "3", "4"]
    labels = ["No Noise"]
    for layer in layer_settings:
            for noise in noise_settings:
                labels.append(f"{noise}, L={layer}")
    # n_colors = len(labels) // 4
    n_colors = 3
    #create color palette
    palette = ["#000000"]
    palette.extend(sns.color_palette(palette="Reds", n_colors=n_colors+1).as_hex()[1:])
    palette.extend(sns.color_palette(palette="Greens", n_colors=n_colors+1).as_hex()[1:])
    palette.extend(sns.color_palette(palette="Blues", n_colors=n_colors+1).as_hex()[1:])        
    palette.extend(sns.color_palette(palette="Purples", n_colors=n_colors+1).as_hex()[1:])
    palette.extend(sns.color_palette(palette="Wistia", n_colors=n_colors).as_hex())

    s = sns.lineplot(
        ax=ax,
        data=df, x="covtrans_scale", y="accuracy",
        errorbar=None,
        hue="label", legend="full",
        hue_order=labels,
        palette=palette,
    )
    
    # Set x
    x_ticks = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    x_label = "Scale"
    ax.set_xticks(x_ticks) 
    ax.set_xlim(x_ticks[0], x_ticks[-1]) 
    s.set_xticklabels(x_ticks) 
    ax.set_xlabel(x_label)

    # Set y
    y_ticks = np.arange(start=0, stop=0.95, step=0.05)
    ax.set_yticks(y_ticks)
    ax.set_ylim(bottom=0, top=0.925)
    y_tick_labels = y_ticks.tolist()
    for j in range(len(y_tick_labels)):

        if j % 2 == 0:
            y_tick_labels[j] = ""
        else:
            y_tick_labels[j] = "%.2f" % y_ticks[j]

    ax.set_ylabel("Test Accuracy")
    ax.set_yticklabels(y_tick_labels)

    fig.subplots_adjust(right=0.85)
    plt.savefig(f"{figures_folder}{title.replace(":", "").replace(",", "").replace(' ','_')}.png")

def old_layers_plot(title, df):
    df = df[df['dataset'] != 'train']
    df["label"] = df["noise"] + ", L=" + (df["noisy_layer"]+1).astype(str)
    df = df.replace(to_replace="No Noise, L=1", value="No Noise")
    df = df.replace(to_replace="No Noise, L=2", value="No Noise")
    df = df.replace(to_replace="No Noise, L=3", value="No Noise")
    df = df.replace(to_replace="No Noise, L=4", value="No Noise")
    df = df.replace(to_replace="No Noise, L=5", value="No Noise")    
    sns.set_style("whitegrid")
    #define dimensions of subplots (rows, columns)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    # create labels:
    noise_settings = ["Full Cov", "Diagonal", "Identity"]
    layer_settings = ["1", "2", "3", "4", "5"]
    labels = []
    for layer in layer_settings:
            for noise in noise_settings:
                labels.append(f"{noise}, L={layer}")
    labels.append("No Noise")
    n_colors = 3
    #create color palette
    palette = []
    palette.extend(["#A20A3A", "#F13030", "#FF89AE", "#A36E05E8", "#FFAD0AE8", "#FFDC52FF"])
    palette.extend(sns.color_palette(palette="Greens", n_colors=n_colors+1).as_hex()[1:][::-1])
    palette.extend(sns.color_palette(palette="Blues", n_colors=n_colors+1).as_hex()[1:][::-1])        
    palette.extend(sns.color_palette(palette="Purples", n_colors=n_colors+1).as_hex()[1:][::-1])   
    palette.append("#000000")
    
    for i in range(3):
        if i == 0: # 
            subplot_title = "AutoPGD Attack, tr=2.0"
            attack = "AutoPGD"
            x = "eps"
            x_ticks = [0.001, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
            x_tick_labels = [0.001, "", 0.04, "", 0.08, "", 0.12, "", 0.16, "", 0.2]
            x_tick_rot = 0
            x_label = "Attack Strength $\epsilon$"

            plot_data = df[df['attack'] == attack]
            plot_data = plot_data[plot_data['trace'] > 1.6]
            plot_data = plot_data[plot_data['covadv'] == attack]
        elif i == 1: 
            subplot_title = "Motion Blur, tr=0.5"
            corruption = "motion_blur"
            x = "corr_sev"
            x_ticks = [1, 2, 3, 4, 5]
            x_tick_labels = x_ticks
            x_tick_rot = 0
            x_label = "Severity"
            plot_data = df[df['attack'] == "transformed"]
            plot_data = plot_data[plot_data['covtrans'] == corruption]
            plot_data = plot_data[plot_data['trace'] < 1.0]


        elif i == 2: 
            subplot_title = "Obstruction, tr=0.5"
            transform = "obstruction"
            x = "covtrans_scale"
            x_ticks = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
            x_tick_labels = ["", 0.4, "", 0.8,"", 1.2, "", 1.6,"", 2.0]
            x_tick_rot = 0
            x_label = "Scale"
            plot_data = df[df['attack'] == "transformed"]
            plot_data = plot_data[plot_data['covtrans'] == transform]
            plot_data = plot_data[plot_data['trace'] < 1.0]


        s = sns.lineplot(
            ax=ax[i],
            data=plot_data, x=x, y="accuracy",
            hue="label", legend=(False if i != 0 else "full"),
            errorbar=None,
            hue_order=labels,
            palette=palette,
        )

        # Use dashed line for no noise         
        lines = ax[i].get_lines()
        lines[15].set_linestyle('--')

        # Change the legend entry to match
        legend = ax[i].get_legend()
        if legend is not None:
            for legline in legend.get_lines():
                if legline.get_label() == "No Noise":
                    legline.set_linestyle("--")

        if i== 0:
            sns.move_legend(ax[i], "upper left", bbox_to_anchor=(3.47, 1.1), title="Noise Cov, Noisy Layer")
        ax[i].set_title(subplot_title)

        # Set y
        y_ticks = np.arange(start=0, stop=0.95, step=0.05)
        ax[i].set_yticks(y_ticks)
        ax[i].set_ylim(bottom=0, top=0.925)
 
        if i == 0:
            y_tick_labels = y_ticks.tolist()
            for j in range(len(y_tick_labels)):

                if j % 2 == 0:
                    y_tick_labels[j] = ""
                else:
                    y_tick_labels[j] = "%.2f" % y_ticks[j]

            ax[i].set_ylabel("Test Accuracy")
            ax[i].set_yticklabels(y_tick_labels)
        else: 
            ax[i].set_ylabel("")
            ax[i].set_yticklabels([""] * len(y_ticks))

        # Set x
        ax[i].set_xticks(x_ticks) 
        ax[i].set_xlim(left=x_ticks[0], right=x_ticks[-1])
        s.set_xticklabels(rotation=x_tick_rot, labels=x_tick_labels) 
        ax[i].set_xlabel(x_label)
    fig.supxlabel("Modification Strength", fontsize="large", y=0.01, x=0.465)
    fig.subplots_adjust(right=0.8, bottom=0.18)
    plt.tight_layout()
    full_title = f'{title}'
    plt.savefig(f"{figures_folder}{full_title.replace(":", "").replace(",", "").replace(' ','_')}.png")

def layers_plot(title, df):
    df = df[df['dataset'] != 'train']
    sns.set_style("whitegrid")
    #define dimensions of subplots (rows, columns)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    # create labels:
    noise_settings = ["Full Cov", "Diagonal", "Identity"]
    strength_settings = ["Low", "Medium", "High"]
    labels = []
    for strength in strength_settings:
            for noise in noise_settings:
                labels.append(f"{noise}, {strength}")
    n_colors = 3
    palette = []
    palette.extend(sns.color_palette(palette="Greens", n_colors=n_colors+1).as_hex()[1:][::-1])
    palette.extend(sns.color_palette(palette="Blues", n_colors=n_colors+1).as_hex()[1:][::-1]) 
    palette.extend(sns.color_palette(palette="Reds", n_colors=n_colors+1).as_hex()[1:][::-1])
  
    def get_label(row, strength_var, low, medium, high):
        if np.isclose(row[strength_var], low):
            return row["noise"] + ", Low"
        elif np.isclose(row[strength_var], medium):
            return row["noise"] + ", Medium"
        elif np.isclose(row[strength_var], high):
            return row["noise"] + ", High"
        return None  # fallback for unexpected values
    
    for i in range(3):
        if i == 0: # 
            subplot_title = "AutoPGD Attack, $tr=2.0$"
            attack = "AutoPGD"
            plot_data = df[df['attack'] == attack]
            plot_data = plot_data[plot_data['trace'] > 1.6]
            plot_data = plot_data[plot_data['covadv'] == attack]
            plot_data["label"] = plot_data.apply(get_label, axis=1, strength_var="eps", low=0.04, medium=0.10, high=0.18)

        elif i == 1: 
            subplot_title = "Motion Blur, $tr=0.5$"
            corruption = "motion_blur"
            plot_data = df[df['attack'] == "transformed"]
            plot_data = plot_data[plot_data['covtrans'] == corruption]
            plot_data = plot_data[plot_data['trace'] < 1.0]
            plot_data["label"] = plot_data.apply(get_label, axis=1, strength_var="corr_sev", low=1, medium=3, high=5)

        elif i == 2: 
            subplot_title = "Obstruction, $tr=0.5$"
            transform = "obstruction"
            plot_data = df[df['attack'] == "transformed"]
            plot_data = plot_data[plot_data['covtrans'] == transform]
            plot_data = plot_data[plot_data['trace'] < 1.0]
            plot_data["label"] = plot_data.apply(get_label, axis=1, strength_var="covtrans_scale", low=0.4, medium=1.0, high=1.8)

        s = sns.lineplot(
            ax=ax[i],
            data=plot_data, x="noisy_layer", y="accuracy",
            hue="label", legend=(False if i != 0 else "full"),
            errorbar="ci",
            hue_order=labels,
            palette=palette,
        )

        # Change line styles
        lines = ax[i].get_lines()
        lines[1].set_linestyle('--')
        lines[4].set_linestyle('--')
        lines[7].set_linestyle('--')
        lines[2].set_linestyle(':')
        lines[5].set_linestyle(':')
        lines[8].set_linestyle(':')

        # Change the legend entry to match
        legend = ax[i].get_legend()
        if legend is not None:
            for legline in legend.get_lines():
                label = legline.get_label()
                if "Diagonal" in label:
                    legline.set_linestyle("--")
                elif "Identity" in label:
                    legline.set_linestyle(":")

        if i== 0:
            sns.move_legend(ax[i], "upper left", bbox_to_anchor=(3.47, 1.0), title="Noise Cov, Modification Strength")
        ax[i].set_title(subplot_title)

        # Set y
        y_ticks = np.arange(start=0, stop=0.95, step=0.05)
        ax[i].set_yticks(y_ticks)
        ax[i].set_ylim(bottom=0, top=0.925)
 
        if i == 0:
            y_tick_labels = y_ticks.tolist()
            for j in range(len(y_tick_labels)):

                if j % 2 == 0:
                    y_tick_labels[j] = ""
                else:
                    y_tick_labels[j] = "%.2f" % y_ticks[j]

            ax[i].set_ylabel("Test Accuracy")
            ax[i].set_yticklabels(y_tick_labels)
        else: 
            ax[i].set_ylabel("")
            ax[i].set_yticklabels([""] * len(y_ticks))

        # Set x
        x_ticks = [0, 1, 2, 3, 4]
        
        ax[i].set_xticks(x_ticks) 
        ax[i].set_xlim(left=x_ticks[0], right=x_ticks[-1])
        s.set_xticklabels(labels=["1", "2", "3", "4", "5"]) 
        ax[i].set_xlabel("")

    fig.supxlabel("Noisy Layer Index $L$", fontsize="large", y=0.05, x=0.416)
    fig.subplots_adjust(right=0.7, bottom=0.18)
    plt.tight_layout()
    full_title = f'{title}'
    plt.savefig(f"{figures_folder}{full_title.replace(":", "").replace(",", "").replace(' ','_')}.png", dpi=200)

def supp_layers_plot(title, df):
    df = df[df['dataset'] != 'train']
    sns.set_style("whitegrid")
    #define dimensions of subplots (rows, columns)
    fig, ax = plt.subplots(5, 3, figsize=(15, 19))
    # create labels:
    noise_settings = ["Full Cov", "Diagonal", "Identity"]
    strength_settings = ["Low", "Medium", "High"]
    labels = []
    for strength in strength_settings:
        for noise in noise_settings:
            labels.append(f"{noise}, {strength}")

    n_colors = 3
    palette = []
    palette.extend(sns.color_palette(palette="Greens", n_colors=n_colors+1).as_hex()[1:][::-1])
    palette.extend(sns.color_palette(palette="Blues", n_colors=n_colors+1).as_hex()[1:][::-1])
    palette.extend(sns.color_palette(palette="Reds", n_colors=n_colors+1).as_hex()[1:][::-1])
    
    def get_label(row, strength_var, low, medium, high):
        if np.isclose(row[strength_var], low):
            return row["noise"] + ", Low"
        elif np.isclose(row[strength_var], medium):
            return row["noise"] + ", Medium"
        elif np.isclose(row[strength_var], high):
            return row["noise"] + ", High"
        return None
    
    row_1 = ["AutoPGD", "PGD"]
    row_2 = ["FGM", "Square", "elastic"]
    row_3 = ["perspective", "obstruction", "rotate"]
    row_4 = ["brightness", "contrast", "gaussian_noise"]
    row_5 = ["impulse_noise", "motion_blur", "snow"]
    rows = [row_1, row_2, row_3, row_4, row_5]

    type_dict = {
        "AutoPGD": "attack",
        "PGD": "attack",
        "FGM": "attack",
        "Square": "attack",
        "elastic": "transform",
        "perspective": "transform",
        "obstruction": "transform",
        "rotate": "transform",
        "brightness": "corruption",
        "contrast": "corruption",
        "gaussian_noise": "corruption",
        "impulse_noise": "corruption",
        "motion_blur": "corruption",
        "snow": "corruption",
    }

    for row in range(5):
        row_content = rows[row]
        for col in range(3):
            if row == 0 and col == 2:
                ax[row][col].set_visible(False)
                continue
            if type_dict[row_content[col]] == "attack":
                attack = row_content[col]
                plot_data = df[df['attack'] == attack].copy()
                plot_data = plot_data[plot_data['covadv'] == attack]
                plot_data = plot_data[plot_data['trace'] < 2.1]
                plot_data = plot_data[plot_data['trace'] > 1.9]
                plot_data["label"] = plot_data.apply(get_label, axis=1, strength_var="eps", low=0.05, medium=0.10, high=0.15,)
                subplot_title = f"{row_content[col]} Attack, $tr=2.0$"

            elif type_dict[row_content[col]] == "transform":
                if row_content[col] == "obstruction":
                    subplot_title = f"Obstruction, $tr=0.5$"
                else:
                    subplot_title = f"{row_content[col]}, $tr=0.5$"
                transform = row_content[col]
                plot_data = df[df['attack'] == "transformed"].copy()
                plot_data = plot_data[plot_data['covtrans'] == transform]
                plot_data = plot_data[plot_data['trace'] < 0.6]
                plot_data = plot_data[plot_data['trace'] > 0.4]
                plot_data["label"] = plot_data.apply(get_label, axis=1, strength_var="covtrans_scale", low=0.5, medium=1.0, high=1.5)

            elif type_dict[row_content[col]] == "corruption":
                name_dict = {
                    "brightness": "Brightness",
                    "contrast": "Contrast",
                    "gaussian_noise": "Gaussian Noise",
                    "impulse_noise": "Impulse Noise",
                    "motion_blur": "Motion Blur",
                    "snow": "Snow",
                }

                subplot_title = f"{name_dict[row_content[col]]}, $tr=0.5$"
                corruption = row_content[col]
                plot_data = df[df['attack'] == "transformed"]
                plot_data = plot_data[plot_data['covtrans'] == corruption]
                plot_data = plot_data[plot_data['trace'] < 0.6]
                plot_data = plot_data[plot_data['trace'] > 0.4]
                plot_data["label"] = plot_data.apply(get_label, axis=1, strength_var="corr_sev", low=1, medium=3, high=5)

            s = sns.lineplot(
                ax=ax[row][col],
                data=plot_data, x="noisy_layer", y="accuracy",
                hue="label", 
                legend=(False if row != 0 or col != 1 else "full"),
                errorbar="ci",
                hue_order=labels,
                palette=palette,
            )

            # Change line styles
            lines = ax[row][col].get_lines()
            lines[1].set_linestyle('--')
            lines[4].set_linestyle('--')
            lines[7].set_linestyle('--')
            lines[2].set_linestyle(':')
            lines[5].set_linestyle(':')
            lines[8].set_linestyle(':')

            legend = ax[row][col].get_legend()
            if legend is not None:
                for legline in legend.get_lines():
                    label = legline.get_label()
                    if "Diagonal" in label:
                        legline.set_linestyle("--")
                    elif "Identity" in label:
                        legline.set_linestyle(":")

            if row== 0 and col == 1:
                sns.move_legend(ax[row][col], "center", bbox_to_anchor=(1.4, 0.5), title="Noise Cov, Modification Strength")
            ax[row][col].set_title(subplot_title)

            # Set y
            y_ticks = np.arange(start=0, stop=0.95, step=0.05)
            ax[row][col].set_yticks(y_ticks)
            ax[row][col].set_ylim(bottom=0, top=0.925)
    
            y_tick_labels = y_ticks.tolist()
            for j in range(len(y_tick_labels)):

                if j % 2 == 0:
                    y_tick_labels[j] = ""
                else:
                    y_tick_labels[j] = "%.2f" % y_ticks[j]

            if col == 0:
                ax[row][col].set_ylabel("Test Accuracy")
            else:
                ax[row][col].set_ylabel("")

            ax[row][col].set_yticklabels(y_tick_labels)
            # Set x
            x_ticks = [0, 1, 2, 3, 4]
            ax[row][col].set_xlim(x_ticks[0], x_ticks[-1])
            ax[row][col].set_xticks(x_ticks) 
            ax[row][col].set_xticklabels(["1", "2", "3", "4", "5"])
            ax[row][col].set_xlabel("")
    fig.supxlabel("Noisy Layer Index $L$", fontsize="large")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2) 
    full_title = f'{title}'
    plt.savefig(f"{figures_folder}{full_title.replace(":", "").replace(",", "").replace(' ','_')}.png", dpi=200)
    
def supp_trace_plot(title, df):
    df = df[df['dataset'] != 'train']
    sns.set_style("whitegrid")
    #define dimensions of subplots (rows, columns)
    fig, ax = plt.subplots(5, 3, figsize=(15, 19))
    # create labels:
    noise_settings = ["Full Cov", "Diagonal", "Identity"]
    strength_settings = ["Low", "Medium", "High"]
    labels = []
    for strength in strength_settings:
        for noise in noise_settings:
            labels.append(f"{noise}, {strength}")

    n_colors = 3
    palette = []
    palette.extend(sns.color_palette(palette="Greens", n_colors=n_colors+1).as_hex()[1:][::-1])
    palette.extend(sns.color_palette(palette="Blues", n_colors=n_colors+1).as_hex()[1:][::-1])
    palette.extend(sns.color_palette(palette="Reds", n_colors=n_colors+1).as_hex()[1:][::-1])
    
    def get_label(row, strength_var, low, medium, high):
        if np.isclose(row[strength_var], low):
            return row["noise"] + ", Low"
        elif np.isclose(row[strength_var], medium):
            return row["noise"] + ", Medium"
        elif np.isclose(row[strength_var], high):
            return row["noise"] + ", High"
        return None
    
    row_1 = ["AutoPGD", "PGD"]
    row_2 = ["FGM", "Square", "elastic"]
    row_3 = ["perspective", "obstruction", "rotate"]
    row_4 = ["brightness", "contrast", "gaussian_noise"]
    row_5 = ["impulse_noise", "motion_blur", "snow"]
    rows = [row_1, row_2, row_3, row_4, row_5]

    type_dict = {
        "AutoPGD": "attack",
        "PGD": "attack",
        "FGM": "attack",
        "Square": "attack",
        "elastic": "transform",
        "perspective": "transform",
        "obstruction": "transform",
        "rotate": "transform",
        "brightness": "corruption",
        "contrast": "corruption",
        "gaussian_noise": "corruption",
        "impulse_noise": "corruption",
        "motion_blur": "corruption",
        "snow": "corruption",
    }

    for row in range(5):
        row_content = rows[row]
        for col in range(3):
            if row == 0 and col == 2:
                ax[row][col].set_visible(False)
                continue
            if type_dict[row_content[col]] == "attack":
                attack = row_content[col]
                plot_data = df[df['attack'] == attack].copy()
                plot_data = plot_data[plot_data['covadv'] == attack]
                plot_data["label"] = plot_data.apply(get_label, axis=1, strength_var="eps", low=0.05, medium=0.10, high=0.15,)
                subplot_title = f"{row_content[col]} Attack"

            elif type_dict[row_content[col]] == "transform":
                if row_content[col] == "obstruction":
                    subplot_title = f"Obstruction"
                else:
                    subplot_title = f"{row_content[col]}"
                transform = row_content[col]
                plot_data = df[df['attack'] == "transformed"].copy()
                plot_data = plot_data[plot_data['covtrans'] == transform]
                plot_data["label"] = plot_data.apply(get_label, axis=1, strength_var="covtrans_scale", low=0.5, medium=1.0, high=1.5)

            elif type_dict[row_content[col]] == "corruption":
                name_dict = {
                    "brightness": "Brightness",
                    "contrast": "Contrast",
                    "gaussian_noise": "Gaussian Noise",
                    "impulse_noise": "Impulse Noise",
                    "motion_blur": "Motion Blur",
                    "snow": "Snow",
                }

                subplot_title = f"{name_dict[row_content[col]]}"
                corruption = row_content[col]
                plot_data = df[df['attack'] == "transformed"]
                plot_data = plot_data[plot_data['covtrans'] == corruption]
                plot_data["label"] = plot_data.apply(get_label, axis=1, strength_var="corr_sev", low=1, medium=3, high=5)

            plot_data['trace'] = plot_data['trace'].astype(str)
            trace_order = ["nan", "0.25", "0.5", "1.0", "2.0"]
            plot_data['trace'] = pd.Categorical(plot_data['trace'], categories=trace_order, ordered=True)

            s = sns.lineplot(
                ax=ax[row][col],
                data=plot_data, x="trace", y="accuracy",
                hue="label", 
                legend=(False if row != 0 or col != 1 else "full"),
                errorbar="ci",
                hue_order=labels,
                palette=palette,
            )

            # Change line styles
            lines = ax[row][col].get_lines()
            lines[1].set_linestyle('--')
            lines[4].set_linestyle('--')
            lines[7].set_linestyle('--')
            lines[2].set_linestyle(':')
            lines[5].set_linestyle(':')
            lines[8].set_linestyle(':')

            legend = ax[row][col].get_legend()
            if legend is not None:
                for legline in legend.get_lines():
                    label = legline.get_label()
                    if "Diagonal" in label:
                        legline.set_linestyle("--")
                    elif "Identity" in label:
                        legline.set_linestyle(":")

            if row== 0 and col == 1:
                sns.move_legend(ax[row][col], "center", bbox_to_anchor=(1.4, 0.5), title="Noise Cov, Modification Strength")
            ax[row][col].set_title(subplot_title)

            # Set y
            y_ticks = np.arange(start=0, stop=0.95, step=0.05)
            ax[row][col].set_yticks(y_ticks)
            ax[row][col].set_ylim(bottom=0, top=0.925)
    
            y_tick_labels = y_ticks.tolist()
            for j in range(len(y_tick_labels)):

                if j % 2 == 0:
                    y_tick_labels[j] = ""
                else:
                    y_tick_labels[j] = "%.2f" % y_ticks[j]

            if col == 0:
                ax[row][col].set_ylabel("Test Accuracy")
            else:
                ax[row][col].set_ylabel("")

            ax[row][col].set_yticklabels(y_tick_labels)
            # Set x
            x_ticks = ["nan", "0.25", "0.5", "1.0", "2.0"]
            ax[row][col].set_xlim(x_ticks[0], x_ticks[-1])
            ax[row][col].set_xticks(x_ticks) 
            ax[row][col].set_xticklabels(["No tr", "0.25", "0.5", "1.0", "2.0"])
            ax[row][col].set_xlabel("")
    fig.supxlabel("Normalized Trace $tr$", fontsize="large")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2) 
    full_title = f'{title}'
    plt.savefig(f"{figures_folder}{full_title.replace(":", "").replace(",", "").replace(' ','_')}.png", dpi=200)

def old_trace_plot(title, df):
    df = df[df['dataset'] != 'train']
    df['trace'] = df['trace'].astype(str)
    df["label"] = df["noise"] + ", tr=" + (df["trace"])
    df = df.replace(to_replace="No Noise, tr=nan", value="No Noise")
    df = df.replace(to_replace="No Noise, tr=0.25", value="No Noise")
    df = df.replace(to_replace="No Noise, tr=0.5", value="No Noise")
    df = df.replace(to_replace="No Noise, tr=1.0", value="No Noise")
    df = df.replace(to_replace="No Noise, tr=2.0", value="No Noise")
    df = df.replace(to_replace="Full Cov, tr=nan", value="Full Cov")
    df = df.replace(to_replace="Diagonal, tr=nan", value="Diagonal")
    df = df.replace(to_replace="Identity, tr=nan", value="Identity")

    sns.set_style("whitegrid")
    #define dimensions of subplots (rows, columns)
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), height_ratios=(2, 1.1))
    # create labels:
    noise_settings = ["Full Cov", "Diagonal", "Identity"]
    trace_settings = ["", ", tr=0.25", ", tr=0.5", ", tr=1.0", ", tr=2.0"]
    labels = []
    for noise in noise_settings:
        for trace in trace_settings:
                labels.append(f"{noise}{trace}")
    labels.append("No Noise")

    n_colors = 5
    #create color palette
    palette = []
    palette.extend(sns.color_palette(palette="Reds", n_colors=n_colors+1).as_hex()[1:])
    palette.extend(sns.color_palette(palette="Greens", n_colors=n_colors+1).as_hex()[1:])
    palette.extend(sns.color_palette(palette="Blues", n_colors=n_colors+1).as_hex()[1:])        
    palette.append("#000000")
    
    for col in range(3):
        for row in range(2):
            if col == 0: # 
                subplot_title = "AutoPGD Attack"
                attack = "AutoPGD"
                x = "eps"
                x_ticks = [0.001, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
                x_tick_labels = [0.001, "", 0.04, "", 0.08, "", 0.12, "", 0.16, "", 0.2]
                x_tick_rot = 0
                x_label = "Attack Strength $\epsilon$"
                plot_data = df[df['covadv'] == attack]
            elif col == 1: 
                subplot_title = "Motion Blur"
                corruption = "motion_blur"
                attack = "transformed"
                x = "corr_sev"
                x_ticks = [1, 2, 3, 4, 5]
                x_tick_labels = x_ticks
                x_tick_rot = 0
                x_label = "Severity"
                plot_data = df[df['covtrans'] == corruption]
            elif col == 2: 
                subplot_title = "Obstruction"
                transform = "obstruction"
                attack = "transformed"
                x = "covtrans_scale"
                x_ticks = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
                x_tick_labels = ["", 0.4, "", 0.8,"", 1.2, "", 1.6,"", 2.0]
                x_tick_rot = 0
                x_label = "Scale"
                plot_data = df[df['covtrans'] == transform]

            if row == 1:
                attack = "benign"
            plot_data = plot_data[plot_data["attack"] == attack]
            s = sns.lineplot(
                ax=ax[row][col],
                data=plot_data, x=x, y="accuracy",
                hue="label", legend=(False if col != 0 or row != 0 else "full"),
                errorbar=None,
                hue_order=labels,
                palette=palette,
            )

            # Use dashed line for no noise         
            lines = ax[row][col].get_lines()
            lines[15].set_linestyle('--')

            # Change the legend entry to match
            legend = ax[row][col].get_legend()
            if legend is not None:
                for legline in legend.get_lines():
                    if legline.get_label() == "No Noise":
                        legline.set_linestyle("--")

            if col == 0 and row == 0:
                sns.move_legend(ax[row][col], "center left", bbox_to_anchor=(3.47, 0.2), title="Noise Cov, tr")

            if row == 0:
                # Set y
                y_ticks = np.arange(start=0, stop=0.95, step=0.05)
                ax[row][col].set_yticks(y_ticks)
                ax[row][col].set_ylim(bottom=0, top=0.925)
            else: 
                subplot_title = f"Clean Data,\n{subplot_title} Covariance"
                y_ticks = np.arange(start=0.7, stop=0.95, step=0.05)
                ax[row][col].set_yticks(y_ticks)
                ax[row][col].set_ylim(bottom=0.68, top=0.925)
            
            if col == 0:
                y_tick_labels = y_ticks.tolist()
                for k in range(len(y_tick_labels)):
                    y_tick_labels[k] = "%.2f" % y_tick_labels[k]
                if row == 0:
                    for k in range(len(y_tick_labels)):

                        if k % 2 == 0:
                            y_tick_labels[k] = ""
                        else:
                            y_tick_labels[k] = "%.2f" % y_ticks[k]

                ax[row][col].set_ylabel("Test Accuracy")
                ax[row][col].set_yticklabels(y_tick_labels)
            else: 
                ax[row][col].set_ylabel("")
                ax[row][col].set_yticklabels([""] * len(y_ticks))

            # Set x
            ax[row][col].set_xticks(x_ticks) 
            ax[row][col].set_xlim(left=x_ticks[0], right=x_ticks[-1])
            if row == 0:
                x_tick_labels = [""] * len(x_tick_labels)
                x_label = ""
            else:
                fig.supxlabel("Modification Strength", fontsize="large", y=0.03, x=0.47)
            s.set_xticklabels(rotation=x_tick_rot, labels=x_tick_labels) 
            ax[row][col].set_xlabel(x_label)
            ax[row][col].set_title(subplot_title)

    fig.subplots_adjust(right=0.8, hspace=0.3, bottom=0.155)
    plt.tight_layout()
    full_title = f'{title}'
    plt.savefig(f"{figures_folder}{full_title.replace(":", "").replace(",", "").replace(' ','_')}.png")

def trace_plot(title, df):
    df = df[df['dataset'] != 'train']
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), height_ratios=(2, 1.1))
    noise_settings = ["Full Cov", "Diagonal", "Identity"]
    strength_settings = ["Low", "Medium", "High"]
    labels = []
    for strength in strength_settings:
        for noise in noise_settings:
            labels.append(f"{noise}, {strength}")

    n_colors = 3
    palette = []
    palette.extend(sns.color_palette(palette="Greens", n_colors=n_colors+1).as_hex()[1:][::-1])
    palette.extend(sns.color_palette(palette="Blues", n_colors=n_colors+1).as_hex()[1:][::-1])
    palette.extend(sns.color_palette(palette="Reds", n_colors=n_colors+1).as_hex()[1:][::-1])

    def get_label(row, strength_var, low, medium, high):
        if np.isclose(row[strength_var], low):
            return row["noise"] + ", Low"
        elif np.isclose(row[strength_var], medium):
            return row["noise"] + ", Medium"
        elif np.isclose(row[strength_var], high):
            return row["noise"] + ", High"
        return None

    for col in range(3):
        for row in range(2):
            if col == 0:
                subplot_title = "AutoPGD Attack"
                attack = "AutoPGD"
                plot_data = df[df['covadv'] == attack].copy()
                plot_data["label"] = plot_data.apply(get_label, axis=1, strength_var="eps", low=0.04, medium=0.10, high=0.18)

            elif col == 1:
                subplot_title = "Motion Blur"
                plot_data = df[df['covtrans'] == "motion_blur"].copy()
                plot_data["label"] = plot_data.apply(get_label, axis=1, strength_var="corr_sev", low=1, medium=3, high=5)

            elif col == 2:
                subplot_title = "Obstruction"
                plot_data = df[df['covtrans'] == "obstruction"].copy()
                plot_data["label"] = plot_data.apply(get_label, axis=1, strength_var="covtrans_scale", low=0.4, medium=1.0, high=1.8)

            attack = "AutoPGD" if col == 0 else "transformed"
            if row == 1:
                attack = "benign"
            plot_data = plot_data[plot_data["attack"] == attack]
            plot_data['trace'] = plot_data['trace'].astype(str)
            trace_order = ["nan", "0.25", "0.5", "1.0", "2.0"]
            plot_data['trace'] = pd.Categorical(plot_data['trace'], categories=trace_order, ordered=True)

            s = sns.lineplot(
                ax=ax[row][col],
                data=plot_data, x="trace", y="accuracy",
                hue="label", legend=(False if col != 0 or row != 0 else "full"),
                errorbar="ci",
                hue_order=labels,
                palette=palette,
            )

            # Change line styles
            lines = ax[row][col].get_lines()
            lines[1].set_linestyle('--')
            lines[4].set_linestyle('--')
            lines[7].set_linestyle('--')
            lines[2].set_linestyle(':')
            lines[5].set_linestyle(':')
            lines[8].set_linestyle(':')

            legend = ax[row][col].get_legend()
            if legend is not None:
                for legline in legend.get_lines():
                    label = legline.get_label()
                    if "Diagonal" in label:
                        legline.set_linestyle("--")
                    elif "Identity" in label:
                        legline.set_linestyle(":")

            if col == 0 and row == 0:
                sns.move_legend(ax[row][col], "center left", bbox_to_anchor=(3.55, 0.2),
                                title="Noise Cov, Modification Strength")

            if row == 0:
                y_ticks = np.arange(start=0, stop=0.95, step=0.05)
                ax[row][col].set_yticks(y_ticks)
                ax[row][col].set_ylim(y_ticks[0], y_ticks[-1])
            else:
                subplot_title = f"Clean Data,\n{subplot_title} Cov"
                y_ticks = np.arange(start=0.7, stop=0.95, step=0.05)
                ax[row][col].set_yticks(y_ticks)
                ax[row][col].set_ylim(bottom=0.68, top=0.925)

            if col == 0:
                y_tick_labels = y_ticks.tolist()
                for k in range(len(y_tick_labels)):
                    y_tick_labels[k] = "%.2f" % y_tick_labels[k]
                if row == 0:
                    for k in range(len(y_tick_labels)):
                        if k % 2 == 0:
                            y_tick_labels[k] = ""
                        else:
                            y_tick_labels[k] = "%.2f" % y_ticks[k]
                ax[row][col].set_ylabel("Test Accuracy")
                ax[row][col].set_yticklabels(y_tick_labels)
                ax[row][col].set_ylim(y_ticks[0], y_ticks[-1])

            else:
                ax[row][col].set_ylabel("")
                ax[row][col].set_yticklabels([""] * len(y_ticks))

            # Set x
            x_ticks = ["nan", "0.25", "0.5", "1.0", "2.0"]
            ax[row][col].set_xlim(x_ticks[0], x_ticks[-1])

            ax[row][col].set_xticks(x_ticks)
            if row == 0:
                ax[row][col].set_xticklabels([""] * len(x_ticks))
                ax[row][col].set_xlabel("")
            else:
                ax[row][col].set_xticklabels(["No tr", "0.25", "0.5", "1.0", "2.0"])
                fig.supxlabel("Normalized Trace $tr$", fontsize="large", y=0.07, x=0.47)
                ax[row][col].set_xlabel("")
            ax[row][col].set_title(subplot_title)

    fig.subplots_adjust(right=0.72, hspace=0.3, bottom=0.155)
    plt.tight_layout()
    full_title = f'{title}'
    plt.savefig(f"{figures_folder}{full_title.replace(':', '').replace(',', '').replace(' ', '_')}.png", dpi=200)

def covadv_equivalent_plot(title, df):
    df = df[df['dataset'] != 'train']
    conditions = [
        df['noise'].isin(['Full Cov', 'Diagonal']),
        df['noise'].isin(['No Noise', 'Identity'])
    ]

    choices = [
        df['noise'] + ", " + df['cov'],
        df['noise']
    ]

    df['label'] = np.select(conditions, choices, default='')

    sns.set_style("whitegrid")
    #define dimensions of subplots (rows, columns)
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.5))

    # create labels:
    cov_settings = ["Gaussian Noise", "AutoPGD", "PGD", "FGM", "Square"]
    noise_settings = ["Full Cov", "Diagonal"]
    labels = []
    for noise in noise_settings:
        for cov in cov_settings:
            labels.append(noise + ", " + cov)
    labels.extend(["Identity", "No Noise"])

    #create color palette
    palette = ["#980129", "#d52221", "#f44f39", "#fc8161", "#dfad07", "#891fcf", "#0b559f", "#457bcc", "#63addb", "#66c093", "#3B3B3B", "#000000"]

    for i in range(2):
        for j in range(2):
            if i == 0 and j == 0:
                subplot_title = "AutoPGD Attack"
                attack = "AutoPGD"
                plot_data = df[df['attack'] == attack]
            elif i == 0 and j == 1:
                subplot_title = "PGD Attack"
                attack = "PGD"
            elif i == 1 and j == 0:
                subplot_title = "FGM Attack"
                attack = "FGM"
            elif i == 1 and j == 1:
                subplot_title = "Square Attack"
                attack = "Square"

            plot_data = df[df['attack'] == attack]
            s = sns.lineplot(
                ax=ax[i][j],
                data=plot_data, x="eps", y="accuracy", errorbar="ci",
                hue="label", legend=(False if i != 0 or j != 1 else "full"),
                hue_order=labels,
                palette=palette,
            )

            # Use dashed line for no noise         
            lines = ax[i][j].get_lines()
            lines[11].set_linestyle('--')

            # Change the legend entry to match
            legend = ax[i][j].get_legend()
            if legend is not None:
                for legline in legend.get_lines():
                    if legline.get_label() == "No Noise":
                        legline.set_linestyle("--")

            if i== 0 and j == 1:
                sns.move_legend(ax[i][j], "center left", bbox_to_anchor=(1.05, 0), title="Noise Cov, Cov Source")
            ax[i][j].set_title(subplot_title)


            # Set y
            y_ticks = np.arange(start=0, stop=0.95, step=0.05)
            ax[i][j].set_yticks(y_ticks)
            ax[i][j].set_ylim(bottom=0, top=0.925)
    
            if j == 0:
                y_tick_labels = y_ticks.tolist()
                for k in range(len(y_tick_labels)):

                    if k % 2 == 0:
                        y_tick_labels[k] = ""
                    else:
                        y_tick_labels[k] = "%.2f" % y_ticks[k]

                ax[i][j].set_ylabel("Test Accuracy")
                ax[i][j].set_yticklabels(y_tick_labels)
            else: 
                ax[i][j].set_ylabel("")
                ax[i][j].set_yticklabels([""] * len(y_ticks))

            # Set x
            x_ticks = [0.001, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
            if i == 0:
                x_tick_labels = [""] * len(x_ticks)
                x_label = ""
            else:
                x_tick_labels = [0.001, "", 0.04, "", 0.08, "", 0.12, "", 0.16, "", 0.2]
                x_label = "Attack Strength $\epsilon$"

            ax[i][j].set_xticks(x_ticks) 
            ax[i][j].set_xlim(left=x_ticks[0], right=x_ticks[-1])
            s.set_xticklabels(labels=x_tick_labels) 
            ax[i][j].set_xlabel(x_label)
            
    fig.supxlabel("Modification Strength", fontsize="large", y=0.03, x=0.44)
    plt.tight_layout()
    fig.subplots_adjust(right=0.75, hspace=0.15)
    full_title = f'{title}'
    plt.savefig(f"{figures_folder}{full_title.replace(":", "").replace(",", "").replace(' ','_')}.png")

def get_dataframe(df_files):
    attack_data = []

    for fp in df_files:
        with open(fp, 'rb') as file:
            this_df = pickle.load(file)
            attack_data.append(this_df)
    df = pd.concat(attack_data)

    return df

def make_table(df, columns, filename):
    # Group by the columns other than 'Accuracy'
    grouped = df.groupby(columns, dropna=False)

    df.groupby(columns)['accuracy'].describe()

    exit()
    print(grouped.size())
    # Calculate mean and standard deviation
    mean_df = grouped.agg(
        mean_accuracy=('accuracy', 'mean'),
        std_accuracy=('accuracy', 'std'),
        count_accuracy=('accuracy', 'count')
    ).reset_index()
    mean_df["lower_ci"] = mean_df["mean"] - 1.96*(mean_df["std"]/np.sqrt(mean_df["count"]))
    mean_df["upper_ci"] = mean_df["mean"] + 1.96*(mean_df["std"]/np.sqrt(mean_df["count"]))

    print(mean_df.head)
    exit()
    columns.extend(["mean_accuracy", "std_accuracy"])
    mean_df.to_excel(f"{figures_folder}{filename}.xlsx", columns=columns, index=False)

#NOTE: Must add a line filtering by transform_scale when you rerun with new data 
def trans_mismatch_table(df, filename):
    df = df[df["trace"] < 0.6]
    df = df[df["corr_sev"] == 5]
    df = df[df["covcorr_sev"] == 5]
    df = df[df["noise"] == "Full Cov"]
    df = df[df["attack"] == "transformed"]
    df.drop(columns=["attack", "noisy_layer", "n_ensemble", "trace", "corr_sev", "covcorr_sev", "noise", "arch", "eps", "ac_eps", "covrot", "covadv", "alpha", "beta"], inplace=True)
    
    ci_df = df.groupby(["covtrans", "transform"])["accuracy"].describe()[["count", "mean", "std"]].reset_index()
    ci_df["lower_ci"] = ci_df["mean"] - 1.96*(ci_df["std"]/np.sqrt(ci_df["count"]))
    ci_df["upper_ci"] = ci_df["mean"] + 1.96*(ci_df["std"]/np.sqrt(ci_df["count"]))
    ci_df["Result"] = (
        ci_df["mean"].apply(lambda x: f"{x:.2f}") +
        " [" +
        ci_df["lower_ci"].apply(lambda x: f"{x:.2f}") + "," +
        ci_df["upper_ci"].apply(lambda x: f"{x:.2f}") +
        "]"
    )
    ci_df.drop(columns=["lower_ci", "upper_ci", "mean", "count", "std"], inplace=True)

    # One column for each covariance, one row for each data alteration 
    transform_names = np.sort(ci_df["transform"].unique()).tolist()
    result = pd.DataFrame(columns=["Distortion"] + transform_names)
    for transform in transform_names:
        row_df = ci_df[ci_df["transform"] == transform]
        assert(len(row_df) == len(transform_names))
        row = [transform] + [row_df[row_df["covtrans"] == covtrans]["Result"].item() for covtrans in transform_names]
        result.loc[len(result)] = row
    result.to_excel(f"{figures_folder}{filename}.xlsx", index=False, header=True)

def supp_covadv_gauss_aug_plot(df, baseline_df, filename, benign=False):
    no_aug_label = "No Aug, Full Cov"
    no_noise_label = "No Noise (only Aug.)"
    df['label'] = np.where(df['noise'] != 'No Noise', df["noise"] + ", tr=" + df["trace"].astype(str), no_noise_label)
    baseline_df['label'] = no_aug_label
    filtered_data = pd.concat([df, baseline_df])
    
    filtered_data = filtered_data[filtered_data['dataset'] != 'train']

    plt.tight_layout()
    sns.set_style("whitegrid")
    #define dimensions of subplots (rows, columns)
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.5))
    # # create labels:
    noise_settings = ["Full Cov", "Diagonal", "Identity"]
    traces = np.sort(filtered_data["trace"].unique())
    labels = []
    for noise in noise_settings:
        for tr in traces:
                labels.append(noise + ", tr=" + tr.astype(str))
    labels.append(no_noise_label)
    labels.append(no_aug_label)
    n_colors = 2

    palette = []
    palette.extend(sns.color_palette(palette="Reds", n_colors=n_colors+1).as_hex()[1:])    
    palette.extend(sns.color_palette(palette="Greens", n_colors=n_colors+1).as_hex()[1:])
    palette.extend(sns.color_palette(palette="Purples", n_colors=n_colors+1).as_hex()[1:])
    palette.append("#000000")
    palette.append("#2E45BB")
    attacks = ["AutoPGD", "PGD", "FGM", "Square"]

    for i in [0, 1]:
        for j in [0, 1]:
            attack = attacks[j + (i*2)]
            if benign:
                plot_data = filtered_data[filtered_data['attack'] == 'benign']
            else:
                plot_data = filtered_data[filtered_data['attack'] == attack]
            plot_data = plot_data[plot_data['covadv'] == attack]

            s = sns.lineplot(
                ax=ax[i, j],
                data=plot_data, x="eps", y="accuracy",
                hue="label", legend=(False if i != 0 or j!=1 else "full"),
                hue_order=labels,
                errorbar="ci",
                palette=palette,
            )

            # Use dashed line for no noise         
            lines = ax[i][j].get_lines()
            lines[6].set_linestyle('--')
            lines[7].set_linestyle('-.')

            # Change the legend entry to match
            legend = ax[i][j].get_legend()
            if legend is not None:
                for legline in legend.get_lines():
                    if legline.get_label() == no_noise_label:
                        legline.set_linestyle("--")
                    elif legline.get_label() == no_aug_label:
                        legline.set_linestyle("-.")
            if i==0 and j == 1:
                sns.move_legend(ax[i,j], "center left", bbox_to_anchor=(1.05, 0), title="Models")
            ax[i,j].set_title(f"{attack}")
        
            y_ticks = np.arange(start=0, stop=0.95, step=0.05)
            ax[i, j].set_yticks(y_ticks)
            ax[i, j].set_ylim(bottom=0, top=0.925)
            x_ticks = [0.001, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
            ax[i, j].set_xticks(x_ticks) 
            ax[i,j].set_xlim(left=0.001, right=0.2)

            if j != 0:
                s.set_yticklabels(labels="") 
                s.set_ylabel("")
            else:
                s.set_ylabel("Test Accuracy")
                y_tick_labels = ticks_to_labels(y_ticks)
                ax[i][j].set_yticklabels(labels=y_tick_labels)
            if i == 0:
                    s.set_xticklabels([""] * len(x_ticks))
                    ax[i][j].set_xlabel("")
            else:
                x_tick_labels = ticks_to_labels(x_ticks)
                s.set_xticklabels(x_tick_labels) 

    plt.tight_layout()
    fig.subplots_adjust(right=0.75, hspace=0.15)
    
    plt.savefig(f"{figures_folder}{filename}.png", dpi=200)

def supp_gauss_aug(df, baseline_df, filename):
    no_aug_label = "No Aug, Full Cov"
    df['label'] = "$\sigma$=" + df["transform_scale"].astype(str)
    baseline_df['label'] = no_aug_label
    filtered_data = pd.concat([df, baseline_df])
    filtered_data = filtered_data[filtered_data['dataset'] != 'train']
    filtered_data = filtered_data[filtered_data['attack'] != 'benign']

    plt.tight_layout()
    sns.set_style("whitegrid")
    
    #define dimensions of subplots (rows, columns)
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.5))

    # create labels:
    traces = np.sort(df["transform_scale"].unique())
    labels = []
    for tr in traces:
            labels.append("$\sigma$=" + tr.astype(str))
    labels.append(no_aug_label)

    #create color palette
    n_colors = 5
    palette = []
    palette.extend(sns.color_palette(palette="Reds", n_colors=n_colors+1).as_hex()[1:])
    palette.append("#2E45BB")
    
    attacks = ["AutoPGD", "PGD", "FGM", "Square"]
    for i in [0, 1]:
        for j in [0, 1]:
            attack = attacks[j + (i*2)]
            plot_data = filtered_data[filtered_data['attack'] == attack]

            s = sns.lineplot(
                ax=ax[i, j],
                data=plot_data, x="eps", y="accuracy", errorbar="ci",
                hue="label", legend=(False if i != 0 or j!=1 else "full"),
                hue_order=labels,
                palette=palette,
            )

            # Use dashed line for no aug       
            lines = ax[i][j].get_lines()
            lines[5].set_linestyle('-.')

            # Change the legend entry to match
            legend = ax[i][j].get_legend()
            if legend is not None:
                for legline in legend.get_lines():
                    if legline.get_label() == no_aug_label:
                        legline.set_linestyle("-.")
            if i==0 and j == 1:
                sns.move_legend(ax[i,j], "center left", bbox_to_anchor=(1.05, 0), title="Models")
            ax[i,j].set_title(f"{attack}")
        
            y_ticks = np.arange(start=0, stop=0.95, step=0.05)
            ax[i, j].set_yticks(y_ticks)
            ax[i, j].set_ylim(bottom=0, top=0.925)
            x_ticks = [0.001, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
            ax[i, j].set_xticks(x_ticks) 
            ax[i,j].set_xlim(left=0.001, right=0.2)

            if j != 0:
                s.set_yticklabels(labels="") 
                s.set_ylabel("")
            else:
                s.set_ylabel("Test Accuracy")
                y_tick_labels = ticks_to_labels(y_ticks)
                ax[i][j].set_yticklabels(labels=y_tick_labels)
            if i == 0:
                    s.set_xticklabels([""] * len(x_ticks))
                    ax[i][j].set_xlabel("")
            else:
                x_tick_labels = ticks_to_labels(x_ticks)
                s.set_xticklabels(x_tick_labels) 

    plt.tight_layout()
    fig.subplots_adjust(right=0.75, hspace=0.15)

    plt.savefig(f"{figures_folder}{filename}.png", dpi=200)

def supp_adv_train(df, baseline_df, filename):
    no_aug_label = "0.0, Full Cov, tr=2.0"
    baseline_df = baseline_df[baseline_df["trace"] > 1.9]
    df['label'] = df["ratio"].astype(str)
    baseline_df['label'] = no_aug_label
    filtered_data = pd.concat([df, baseline_df])
    filtered_data = filtered_data[filtered_data['dataset'] != 'train']

    sns.set_style("whitegrid")
    #define dimensions of subplots (rows, columns)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    # # create labels:
    traces = np.sort(df["ratio"].unique())
    labels = []
    for tr in traces:
            labels.append(tr.astype(str))
    labels.append(no_aug_label)
    n_colors = 9
    #create color palette
    palette = []
    palette.extend(sns.color_palette(palette="Reds", n_colors=n_colors+1).as_hex()[1:])

    palette.append("#2E45BB")
    attacks = ["PGD", "benign"]
    for i in [0, 1]:
        attack = attacks[i]
        plot_data = filtered_data[filtered_data['attack'] == attack]

        s = sns.lineplot(
            ax=ax[i],
            data=plot_data, x="eps", y="accuracy", errorbar="ci",
            hue="label", legend=(False if i == 0 else "full"),
            hue_order=labels,
            palette=palette,
        )

        # Use dashed line for no noise         
        lines = ax[i].get_lines()
        lines[9].set_linestyle('-.')

        # Change the legend entry to match
        legend = ax[i].get_legend()
        if legend is not None:
            for legline in legend.get_lines():
                if legline.get_label() == no_aug_label:
                    legline.set_linestyle("-.")
        if i==1:
            sns.move_legend(ax[i], "center left", bbox_to_anchor=(1.05, 0.5), title="Ratio of Adv. Examples")
        ax[i].set_title(f"{attack}")
    
        y_ticks = np.arange(start=0, stop=0.95, step=0.05)
        ax[i].set_yticks(y_ticks)
        ax[i].set_ylim(bottom=0, top=0.925)
        x_ticks = [0.001, 0.05, 0.1, 0.15, 0.2]
        ax[i].set_xticks(x_ticks) 
        ax[i].set_xlim(left=0.001, right=0.2)

        if i != 0:
            s.set_yticklabels(labels="") 
            s.set_ylabel("")
        else:
            s.set_ylabel("Test Accuracy")
            y_tick_labels = y_ticks.tolist()
            for k in range(len(y_tick_labels)):

                if k % 2 == 0:
                    y_tick_labels[k] = ""
                else:
                    y_tick_labels[k] = "%.2f" % y_ticks[k]
            ax[i].set_yticklabels(labels=y_tick_labels)

        s.set_xticklabels(x_ticks) 

    plt.tight_layout()
    fig.subplots_adjust(right=0.75, hspace=0.15)

    plt.savefig(f"{figures_folder}{filename}.png", dpi=200)

def training_loss_plot(title, df):
    """
    df must have columns: ['mb', 'loss', 'epoch', 'file_type', 'modification', 'strength', 'trace']
    file_type values: 'base_model', 'no_noise', 'diagonal_noise', 'identity_noise', 'full_noise'
    'loss' is the raw running loss.
    'mb' is the minibatch number.
    'epoch' is the epoch number.
    """
    df = df.copy()
    label_map = {
        "base_model":     "Base Model",
        "no_noise":       "No Noise",
        "diagonal_noise": "Diagonal",
        "identity_noise": "Identity",
        "full_noise":     "Full Cov",
    }
    df["label"] = df["file_type"].map(label_map)
    palette = {
        "Base Model": "#E24B4A",
        "No Noise":   "#888780",
        "Diagonal":   "#639922",
        "Identity":   "#2E7DC8",
        "Full Cov":   "#7F77DD",
    }
    hue_order = ["Base Model", "Full Cov", "Diagonal", "Identity", "No Noise"]
    # For each epoch, find the last mb value within that epoch
    df = df.sort_values(["epoch", "mb"])
    mb_per_epoch = df.groupby("epoch")["mb"].max()
    epoch_offset = mb_per_epoch.cumsum().shift(1, fill_value=0)
    df["mb_cumulative"] = df["epoch"].map(epoch_offset) + df["mb"]
    epoch_tick_map = (
        df.groupby("epoch")["mb_cumulative"]
        .max()
        .sort_index()
    )
    tick_positions = epoch_tick_map.values
    print("TICK POSITIONS", tick_positions)
    tick_labels = epoch_tick_map.index.tolist()
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(5, 3, figsize=(15, 19))

    row_1 = ["AutoPGD", "PGD"]
    row_2 = ["FGM", "Square", "elastic"]
    row_3 = ["perspective", "obstruction", "rotate"]
    row_4 = ["brightness", "contrast", "gaussian_noise"]
    row_5 = ["impulse_noise", "motion_blur", "snow"]
    rows = [row_1, row_2, row_3, row_4, row_5]

    type_dict = {
        "AutoPGD": "attack",
        "PGD": "attack",
        "FGM": "attack",
        "Square": "attack",
        "elastic": "transform",
        "perspective": "transform",
        "obstruction": "transform",
        "rotate": "transform",
        "brightness": "corruption",
        "contrast": "corruption",
        "gaussian_noise": "corruption",
        "impulse_noise": "corruption",
        "motion_blur": "corruption",
        "snow": "corruption",
    }
    for row in range(5):
        row_content = rows[row]
        for col in range(3):
            if row == 0 and col == 2:
                ax[row][col].set_visible(False)
                continue
            if type_dict[row_content[col]] == "attack":
                subplot_title = f"{row_content[col]} Attack, $\epsilon=0.1$"

            elif type_dict[row_content[col]] == "transform":
                if row_content[col] == "obstruction":
                    subplot_title = f"Obstruction, Scale=1.0"
                else:
                    subplot_title = f"{row_content[col]}, Scale=1.0"

            elif type_dict[row_content[col]] == "corruption":
                name_dict = {
                    "brightness": "Brightness",
                    "contrast": "Contrast",
                    "gaussian_noise": "Gaussian Noise",
                    "impulse_noise": "Impulse Noise",
                    "motion_blur": "Motion Blur",
                    "snow": "Snow",
                }
                subplot_title = f"{name_dict[row_content[col]]}, Severity=3"
            plot_data = df[df["modification"] == row_content[col]]
            sns.lineplot(
                ax=ax[row][col],
                data=plot_data,
                x="mb_cumulative",
                y="loss",
                hue="label",
                hue_order=hue_order,
                palette=palette,
                errorbar="ci",
                legend=(False if row != 0 or col != 1 else "full"),
            )

            if row== 0 and col == 1:
                sns.move_legend(ax[row][col], "center", bbox_to_anchor=(1.3, 0.5), title="Model")
            ax[row][col].set_title(f"Cov Source: {subplot_title}")
            ax[row][col].set_xticks(tick_positions)
            ax[row][col].set_xticklabels([str(int(label)) for label in tick_labels])
            ax[row][col].set_xlim(left=0, right=int(df["mb_cumulative"].max()))
            ax[row][col].set_xlabel("Epoch")
            ax[row][col].set_ylabel("Loss")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    save_title = title.replace(":", "").replace(",", "").replace(" ", "_")
    plt.savefig(f"{figures_folder}{save_title}.png", dpi=200)
    plt.close()

def save_clean_df(df, experiment_name):
    cleaned_df = df.replace("RandObstructionB", "obstruction")
    cleaned_df = cleaned_df.replace("attack_use_train ", "eval_on_train") 
    cleaned_df = cleaned_df.rename(columns={"corruption_severity": "corr_sev"})
    cleaned_df.to_pickle(f"../saved/_raw_results/{experiment_name}.pkl")

def main():
    # IT WORKS TABLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    it_works_df = pd.read_pickle("../saved/it_works/robustness_data.pkl")
    make_it_works_table(it_works_df, "it_works_table")

    # IT WORKS PLOT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # it_works_df = pd.read_pickle("../saved/_raw_results/it_works_plot.pkl")
    # it_works_plot("it_works", it_works_df)

    # TRACE PLOT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # trace_df = pd.read_pickle("../saved/_raw_results/trace.pkl")
    # trace_plot("trace", trace_df)

    # LAYERS PLOT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # layers_df = pd.read_pickle("../saved/_raw_results/layers.pkl")
    # layers_plot("layers", layers_df)

    # ADV COV ISNT SPECIAL PLOT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # covadv_equivalent_df = pd.read_pickle("../saved/_raw_results/covadv_equivalent.pkl")
    # covadv_equivalent_plot("covadv_equivalent", covadv_equivalent_df)
    # make_table(covadv_equivalent_df, ["cov", "noise", "eps","attack"], "table_covadv_equivalent")

    # Mismatch mod heatmap ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # mismatch_mod_df = pd.read_pickle("../saved/mismatch_mod/robustness_data.pkl")
    # mismatch_mod_hm(mismatch_mod_df, "mismatch_mod_hm")

    # SUPPLEMENT PLOTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # supp_layers_df = pd.read_pickle("../saved/_raw_results/supp_layers.pkl")
    # supp_layers_plot("supp_layers_plot", supp_layers_df)

    # supp_trace_df = pd.read_pickle("../saved/_raw_results/supp_trace.pkl")
    # supp_trace_plot("supp_trace_plot", supp_trace_df)

    # baseline_df = pd.read_pickle("../saved/_raw_results/baseline.pkl")
    # supp_covadv_gauss_aug_df = pd.read_pickle("../saved/_raw_results/supp_covadv_gauss_aug.pkl")
    # supp_covadv_gauss_aug_plot(supp_covadv_gauss_aug_df, baseline_df, "supp_covadv_gauss_aug")
     
    # supp_gauss_aug_df = pd.read_pickle("../saved/_raw_results/supp_gauss_aug_df.pkl")
    # supp_gauss_aug(supp_gauss_aug_df, baseline_df, "supp_gauss_aug")
    
    # adv_train_df = pd.read_pickle("../saved/_raw_results/adv_train.pkl")
    # pgd_baseline_df = pd.read_pickle("../saved/_raw_results/pgd_baseline.pkl")
    # supp_adv_train(adv_train_df, pgd_baseline_df, "supp_adv_train")

    # training_loss_df = pd.read_pickle("../saved/_raw_results/training_loss.pkl")
    # training_loss_plot("supp_training_curve", training_loss_df)

    # Filtering dataframes and getting confidence intervals 

    # Adv cov transferability experiment numbers
    # df = covadv_equivalent_df.copy()
    # noise = "Full Cov"
    # adv_cov = "AutoPGD"
    # eps = 0.18
    # df_filtered = df[
    #     (df["cov"] == adv_cov) &
    #     (np.isclose(df["eps"], eps)) &
    #     (df["noise"] == noise) &
    #     (df["attack"] == "AutoPGD")
    # ]

    # acc = df_filtered["accuracy"].dropna().values
    # print(acc)
    # n_boot = 10000
    # boot_means = []

    # for _ in range(n_boot):
    #     sample = np.random.choice(acc, size=len(acc), replace=True)
    #     boot_means.append(sample.mean())

    # ci = np.percentile(boot_means, [2.5, 97.5])

    # print("adv cov: ", adv_cov)
    # print("eps: ", eps)
    # print("Mean accuracy:", acc.mean())
    # print("95% bootstrap CI:", ci)

    # # Getting trace eperiment numbers
    # df = trace_df.copy()
    # noise = "Full Cov"
    # trace = 0.5
    # df_filtered = df[
    #     (df["transform"] == "motion_blur") &
    #     (df["covcorr_sev"] == 3) &
    #     (np.isclose(df["trace"], trace)) &
    #     (df["noise"] == noise) &
    #     (df["attack"] == "transformed")

    # ]

    # acc = df_filtered["accuracy"].dropna().values
    # # print(acc)
    # n_boot = 10000
    # boot_means = []

    # for _ in range(n_boot):
    #     sample = np.random.choice(acc, size=len(acc), replace=True)
    #     boot_means.append(sample.mean())

    # ci = np.percentile(boot_means, [2.5, 97.5])

    # print("Trace: ", trace)
    # print("Mean accuracy:", acc.mean())
    # print("95% bootstrap CI:", ci)


    # # Getting layers experiment numbers 
    # df = supp_layers_df.copy()
    # noise = "Identity"
    
    # df_filtered = df[
    #     (df["attack"] == "AutoPGD") &
    #     (np.isclose(df["eps"], 0.1)) &
    #     (np.isclose(df["trace"], 2.0)) &
    #     (df["noisy_layer"] == 1) &
    #     (df["noise"] == noise)
    # ]

    # acc = df_filtered["accuracy"].dropna().values

    # n_boot = 10000
    # boot_means = []

    # for _ in range(n_boot):
    #     sample = np.random.choice(acc, size=len(acc), replace=True)
    #     boot_means.append(sample.mean())

    # ci = np.percentile(boot_means, [2.5, 97.5])

    # print("Noise: ", noise)
    # print("Mean accuracy:", acc.mean())
    # print("95% bootstrap CI:", ci)

if __name__ == "__main__":
    main()