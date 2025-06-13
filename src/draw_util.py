import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

def draw_scatter(x, y, xlabel, ylabel, info=[],
                 xlog=True,
                 ylog=True):

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    x = np.array(x)
    y = np.array(y)
    if xlog:
        x = np.log10(x)
    if ylog:
        y = np.log10(y)
    plt.scatter(x, y)

    text = ' '.join(info)
    plt.text(2.5, 1.3, text)

    plt.show()


def put_into_bins(data, bin_size=100):

    data_size = len(data)
    num_bins = int(data_size/bin_size)

    bins = []

    for b in range(num_bins):
        bin_ = []
        for i in range(bin_size):
            bin_.append(data[b*bin_size+i])
        bins.append(bin_)

    return bins


def binize(data, bin_size=100):
    """
    Put the input data into bins and take their means.
    In: data in list, bin_size (number of data in a bin)
    Out: np array containing the means of the binned data in the bin.
         (1, bin_size)
    """

    binned_data = put_into_bins(data, bin_size)
    binned_data = np.array(binned_data)
    binned_mean_data = np.mean(binned_data, axis=1)
   
    return binned_mean_data


def select_range(data, max_index, min_index=1):

    data = data[min_index:max_index]

    return data
