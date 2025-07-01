import sys, codecs
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import util
import draw_util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('-f', '--freq_threshold', type=int, default=100)
    parser.add_argument('-b', '--bin_size', type=int, default=100)
    parser.add_argument('-m', '--max_rank', type=int, default=20000)
   
    args = parser.parse_args()

    return args


def select_range(data, max_index, min_index=0):

    data = data[min_index:max_index]

    return data


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


def regression(x, y, xlog=True, ylog=True):
    """
    In: x (numpy array): input data
        y (numpy array): output data 
    """
    model = LinearRegression()

    if xlog:
        x = np.log10(x)
    if ylog:
        y = np.log10(y)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    model.fit(x, y)
    coef = model.coef_
    y_hat = model.predict(x)
    sse = np.sum((y - y_hat)**2, axis=0)
    sse = sse / (x.shape[0] - x.shape[1] - 1.0)
    s = np.linalg.inv(np.dot(x.T, x))
    std_err = np.sqrt(np.diagonal(sse*s))
    t = coef / std_err

    score = model.score(x, y)

    return model, score


def main():
    args = parse_args()
        

    token_kappa_freqs = util.load_kappa_freq_file(args.input_file,
                                                  freq_threshold=args.freq_threshold)

    token_kappa_freqs =\
        sorted(token_kappa_freqs, key=lambda x: x[-1], reverse=True)

    max_rank = min(args.max_rank, len(token_kappa_freqs))
    freqs = [ a[2] for a in token_kappa_freqs ]
    freqs  = select_range(freqs, max_index=max_rank)
    vs = [ 1/float(a[1]) for a in token_kappa_freqs ]
    vs = select_range(vs, max_index=max_rank)
    binned_mean_freqs = binize(freqs, bin_size=args.bin_size)
    binned_mean_vs = binize(vs, bin_size=args.bin_size)

    model, score = regression(binned_mean_freqs, binned_mean_vs)

    coef = np.round(model.coef_[0][0], 3)
    score = np.round(score, 2)

    text = args.input_file +r': $\delta=$'
    text +=  str(coef)
    text += r', $R^2=$' + str(score)
    draw_util.draw_scatter(binned_mean_freqs,
                           binned_mean_vs,
                           xlabel=r'$\log_{10}(f)$',
                           ylabel=r'$\log_{10}(v)$',
                           info=[text])

    plt.legend()
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right')
    plt.show()

if __name__ == '__main__':
    main()
