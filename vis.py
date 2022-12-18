import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import ndarray


def heatmap(data, verbose=False):
    fig, axs = plt.subplots(len(data) + 1)
    fig.suptitle('Weights')
    min = np.amin(data)
    max = np.amax(data)
    # sns.heatmap(data[1], ax=axs[0], linewidth=0.5, vmin=0, vmax=max)#center=0)#,vmin=-1, vmax=1)#
    # sns.heatmap(data[1], ax=axs[1], linewidth=0.5, vmin=min, vmax=0)#center=0)#,vmin=-1, vmax=1)#
    for i in range(len(data)):
        if verbose: print(data[i])
        sns.heatmap(data[i], ax=axs[i], linewidth=0.5, vmin=min, vmax=max)  # center=0)#,vmin=-1, vmax=1)#
        axs[i].set_title(i)
    plt.show()
    # fig, axs = plt.subplots(len(data))
    # fig.suptitle('Vertically stacked subplots')
    # for i in range(len(data)):
    # ax = sns.heatmap(data, linewidth=0.5)

def load_history(filepath):
    data = np.load(filepath, allow_pickle=True)
    history_epochs = len(data[0].history['loss'])
    histories_length = len(data)
    data_shape = (histories_length,history_epochs)
    avg = {"loss": np.zeros(history_epochs), "val_loss": np.zeros(history_epochs)}
    losses: ndarray = np.zeros(data_shape)
    val_losses = np.zeros(data_shape)
    
    for i in range(histories_length):
        d = data[i]
        l = np.array(d.history['loss'])
        v_l = np.array(d.history['val_loss'])
        avg['loss'] += l / histories_length
        avg['val_loss'] += v_l / histories_length
        losses[i]=l
        val_losses[i]=v_l
    
    return np.transpose(losses), np.transpose(val_losses), avg

def plot_history(filenames, plt, color1="red",color2="blue"):
    for i in range(len(filenames)):
        l, v, a = load_history(filenames[i])
        # plt.plot(l, alpha=alpha, color="cyan")
        # plt.plot(v, alpha=alpha, color="saddlebrown")
        linestyle = '-'
        if i < 1:
            linestyle = '--'
        plt.plot(a['loss'], color=color2, alpha=1-(0.2*i),linestyle=linestyle)
        plt.plot(a['val_loss'], color=color1, alpha=1-(0.2*i),linestyle=linestyle)


def plot():
    history1 = np.load('histories/nn_1_16_history.npy', allow_pickle='TRUE').item()
    history2 = np.load('histories/nn_10_32_history.npy', allow_pickle='TRUE').item()

    # # summarize history for accuracy
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # summarize history for loss
    plt.plot(history1['loss'])
    plt.plot(history1['val_loss'])
    plt.plot(history2['loss'])
    plt.plot(history2['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['h1train', 'h1test','h2train', 'h2test'], loc='upper left')
    plt.yscale("log")
    plt.show()