import matplotlib.pyplot as plt


def plotthisformeletmego(title, xlabel, ylabel, training_plot, testing_plot, filename, fignum=1):
    legend =[]
    if(filename is None):
        filename = title + ".png"
    fig = plt.figure(fignum)
    fig.suptitle(title)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    if(training_plot is not None):
        plt.plot(training_plot, 'g', label='Training Error')
        legend.append('training error')
    if(testing_plot is not None):
        plt.plot(testing_plot, 'r', label='Testing Error')
        legend.append('testing error')
    
    plt.legend(legend, loc='upper right')
    fig.savefig(filename)