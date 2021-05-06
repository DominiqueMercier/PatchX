import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_series_and_dist(sid, data, label, pred, dists, patch_params=None, patch=None, save=None):
    """Plots the timeseries and the distribution of the classes within the series.

    Args:
        sid (int): sample id
        data (arr): dat of the sample
        label (int): ground truth (sparse)
        pred (int): prediction (sparse)
        dists (arr): distribution of the predicted labels
        patch_params (arr, optional): parameters of the patch. Defaults to None.
        patch (int, optional): id of the patch. Defaults to None.
        save (str, optional): path to save the plot. Defaults to None.
    """
    _, axes = plt.subplots(figsize=(20, 5), nrows=1, ncols=2)
    ax = axes.flat

    # sample
    ax[0].set_title('Sample ID: ' + str(sid) + ' | Label: ' +
                    str(label) + ' | Pred: ' + str(pred))
    ax[0].set_xlabel('Timesteps')
    ax[0].set_ylabel('Value')
    ax[0].plot(data)

    # dist
    bins = np.arange(len(dists))
    bars = np.max(dists, axis=1)
    bar_labels = np.argmax(dists, axis=1)

    if dists.shape[1] > 10:
        cm = plt.get_cmap('tab20')
        cNorm = matplotlib.colors.Normalize(
            vmin=0, vmax=dists.shape[1]-1)
        scalarMap = matplotlib.cm.ScalarMappable(
            norm=cNorm, cmap=cm)
        bar_colors = [scalarMap.to_rgba(c) for c in bar_labels]
    else:
        bar_colors = np.array(['C' + str(c) for c in bar_labels])
    ax[1].set_xlabel('Patches')
    ax[1].set_ylabel('Classification value')

    ax[1].set_title('Class distribution')

    rects = ax[1].bar(bins, bars, color=bar_colors)

    def autolabel(ax, rects, vals):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            ax.annotate('{}'.format(vals[i]),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(ax[1], rects, bar_labels)

    if save is not None:
        fname = 'Series_and_dist_' + str(sid)
        plt.savefig(os.path.join(save, fname), dpi=300,
                    bbox_inches='tight', pad_inches=0.1)

    plt.show()


def plot_patch_and_dist(sid, data, label, pred, dists, patch_params, patch, save=None):
    """plots the patch and the distribution of the classes for the patch.

    Args:
        sid (int): sample id
        data (arr): dat of the sample
        label (int): ground truth (sparse)
        pred (int): prediction (sparse)
        dists (arr): distribution of the predicted labels
        patch_params (arr, optional): parameters of the patch
        patch (int, optional): id of the patch
        save (str, optional): path to save the plot. Defaults to None.
    """
    _, axes = plt.subplots(figsize=(20, 5), nrows=1, ncols=2)
    ax = axes.flat

    # sample
    ax[0].set_title('Sample ID: ' + str(sid) + ' | Label: ' + str(label) + ' | Pred: ' + str(pred) +
                    ' | Patch: ' + str(patch) + ' | Start: ' + str(patch_params[1]) + ' | End: ' + str(patch_params[2]))
    ax[0].axvspan(patch_params[1], patch_params[2], color='grey', alpha=0.5)
    ax[0].set_xlabel('Timesteps')
    ax[0].set_ylabel('Value')
    ax[0].plot(data)

    # dist
    bins = np.arange(dists.shape[0])

    if len(bins) > 10:
        cm = plt.get_cmap('tab20')
        cNorm = matplotlib.colors.Normalize(
            vmin=0, vmax=len(bins)-1)
        scalarMap = matplotlib.cm.ScalarMappable(
            norm=cNorm, cmap=cm)
        bar_colors = [scalarMap.to_rgba(c) for c in bins]
    else:
        bar_colors = np.array(['C' + str(c) for c in bins])

    ax[1].set_xlabel('Classes')
    ax[1].set_ylabel('Softmax')
    ax[1].set_ylim(0, 1)

    ax[1].set_title('Class distribution')
    ax[1].set_xticks(bins)
    ax[1].bar(bins, dists, color=bar_colors)

    if save is not None:
        fname = 'Patch_and_dist_' + str(sid) + '_' + str(patch)
        plt.savefig(os.path.join(save, fname), dpi=300,
                    bbox_inches='tight', pad_inches=0.1)

    plt.show()


def plot_class_overlay(sid, data, label, pred, patch_dists, param_list, only_classes=None, save=None):
    """Plots the class overlay for a given sample usitlizing the patch predictions.

    Args:
        sid (int): sample id
        data (arr): dat of the sample
        label (int): ground truth (sparse)
        pred (int): prediction (sparse)
        patch_dists (arr): distribution of the predicted labels
        patch_params (arr, optional): parameters of the patch. Defaults to None.
        only_classes (list, optional): Filter classes to only use those included in this list. Defaults to None.
        save (str, optional): path to save the plot. Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(20, 5))
    fig.suptitle('Class distribtuion')

    # sample
    ax.set_title('Sample ID: ' + str(sid) + ' | Label: ' +
                 str(label) + ' | Pred: ' + str(pred))
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Value')
    ax.plot(data)

    # dist
    alpha = np.max(patch_dists, axis=1)
    color = np.argmax(patch_dists, axis=1)

    # color range
    if patch_dists.shape[1] > 10:
        cm = plt.get_cmap('tab20')
        cNorm = matplotlib.colors.Normalize(
            vmin=0, vmax=patch_dists.shape[1]-1)
        scalarMap = matplotlib.cm.ScalarMappable(
            norm=cNorm, cmap=cm)
        color_val = [scalarMap.to_rgba(c)
                     for c in np.arange(patch_dists.shape[1])]
    else:
        color_val = np.array(['C' + str(c)
                              for c in np.arange(patch_dists.shape[1])])

    for c in np.unique(color):
        if only_classes is not None:
            if not c in only_classes:
                continue
        ax.axvspan(0, 0, color=color_val[c], label='Class: ' + str(c))

    def bracket(ax, i, pos=[0, 0], scalex=1, scaley=1, text="", textkw={}, linekw={}):
        x = np.array([0, 0.05, 0.45, 0.5])
        y = np.array([0, -0.01, -0.01, -0.02])
        x = np.concatenate((x, x+pos[1]-pos[0]-0.55))
        y = np.concatenate((y, y[::-1]))
        ax.plot(x*scalex+pos[0], y*scaley-0.12-0.07*(i % 3), clip_on=False,
                transform=ax.get_xaxis_transform(), **linekw)
        ax.text(pos[0]+(pos[1]-pos[0])/2+0.5*scalex, (y.min()-0.01)*scaley-0.12-0.07*(i % 3), text,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", **textkw)

    for i in range(color.shape[0]):
        bracket(ax, i, text='Patch: ' + str(i),
                pos=[param_list[i][1], param_list[i][2]], linekw=dict(color="grey", lw=1))
        if only_classes is not None:
            if not color[i] in only_classes:
                continue
        # , label='Patch: ' + str(i))
        ax.axvspan(param_list[i][1], param_list[i][2],
                   color=color_val[color[i]], alpha=0.25*alpha[i])

    plt.legend()

    if save is not None:
        fname = 'Class_overlay_' + str(sid)
        if only_classes is not None:
            fname += '_c'
            for c in only_classes:
                fname += '-' + str(c)
        plt.savefig(os.path.join(save, fname), dpi=300,
                    bbox_inches='tight', pad_inches=0.1)

    plt.show()


def plot_heatmap(data, class_label=None, save=None):
    """Plots the heatmap plot.

    Args:
        data (arr): dataset array
        class_label (arr, optional): class label array. Defaults to None.
        save (str, optional): path to save the image. Defaults to None.
    """
    feature_label = np.arange(data.shape[1])
    if class_label is None:
        class_label = feature_label

    _, ax = plt.subplots(figsize=(20, 20))

    ax.set_title('Class distribution')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Samples')
    ax.set_xticks(np.arange(len(feature_label)))
    ax.set_yticks(np.arange(len(class_label)))
    ax.set_xticklabels(feature_label)
    ax.set_yticklabels(class_label)

    _ = ax.imshow(data, cmap='cool')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = int(data[i, j] / sum(data[i]) * 100) / 100
            _ = ax.text(j, i, val,
                        ha="center", va="center", color="black")

    if not save is None:
        fname = 'Heatmap'
        plt.savefig(os.path.join(save, fname), dpi=300,
                    bbox_inches='tight', pad_inches=0.1)

    plt.show()


def plot_class_means(class_means, class_label=None, label=True, save=None):
    """Plots the class means plot.

    Args:
        class_means (arr): means of the classes
        class_label (arr, optional): class labels. Defaults to None.
        label (bool, optional): flag to include labels. Defaults to True.
        save (str, optional): path to save the plot. Defaults to None.
    """
    def autolabel(rects, data):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            val = int(data[i] / sum(data) * 100) / 100
            ax.annotate(str(val),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(3, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    bins = np.arange(class_means.shape[1])
    if class_label is None:
        class_label = bins

    width = 0.1

    _, ax = plt.subplots(figsize=(20, 5))

    ax.set_title('Scores by Class and Feature')
    ax.set_ylabel('Distribution')
    ax.set_ylabel('Value')
    ax.set_xticks(bins)
    ax.set_xticklabels(['F' + str(i) for i in bins])

    for i in range(class_means.shape[0]):
        center_off = width * \
            (class_means.shape[0] // 2) - (width / 2) * \
            (1 - class_means.shape[0] % 2)
        rects = ax.bar(bins - center_off + width*i,
                       class_means[i], width, label='Class ' + str(class_label[i]))
        if label:
            autolabel(rects, class_means[i])

    ax.legend()

    if save is not None:
        fname = 'Class_means'
        plt.savefig(os.path.join(save, fname), dpi=300,
                    bbox_inches='tight', pad_inches=0.1)

    plt.show()
