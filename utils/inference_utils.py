import matplotlib.pyplot as plt
import numpy as np


'''
    inference needs to be of the form:
    [   { 'module': module_object , 'shape': numel, 'nonzero': nonzero_count  }          ]
'''


def get_nonzeros_from_inference(inference, ks):
    nonzero = {k:{} for k in ks}
    ii = 0
    for inference in inferences:
        k = ks[ii]
        ii += 1
        for iterdict in inference:
            module = iterdict['module']
            if module not in nonzero[k].keys():
                nonzero[k][module] = [None]*tot_iters
            nonzero[k][module][iterdict['i']] = iterdict['nonzero']
    return nonzero


def fill_axes(ax, xvals, yvals, k, modindex):
    ax.plot(xvals, yvals)
    ax.set_title('perc. sp. k = {}, lay = {}'.format(k, modindex))
 



if __name__ == '__main__':

    ks = [0.1, 0.01, 0.001, 0.0001]
    inferences = []
    start_iter = 0
    end_iter = 40
    tot_iters =  end_iter - start_iter

    modules = exp.modules

    nonzero = get_nonzeros_from_inference(inferences, ks)
    nonzero_percent = [ x for x in nonzero ]



    fig, ax = plt.subplots(len(ks), len(modules), figsize=(15,15))
    fig.tight_layout(pad = 4.0)

    kindex = 0
    for k in ks:
        modindex = 0
        for values in nonzero_percent[k].values():
            fill_axes(ax[kindex][modindex], list(range(len(values))), values, k, modindex)
            ax[kindex][modindex].set_ylim([0. ,1.])
            ax[kindex][modindex].set_xlabel('iters')
            ax[kindex][modindex].set_ylabel('p')

            modindex += 1
        kindex += 1


    fig2, ax2 = plt.subplots(len(ks), len(modules), figsize=(15,15))
    fig2.tight_layout(pad = 4.0)

    kindex = 0
    for k in ks:
        modindex = 0
        for values in nonzero[k].values():
            fill_axes(ax2[kindex][modindex], list(range(len(values))), values, k, modindex)
            fill_axes(ax2[kindex][modindex], list(range(len(values))), np.array(values)/10., k, modindex)
            ax[kindex][modindex].set_xlabel('iters')
            ax[kindex][modindex].set_ylabel('count')

            modindex += 1
        kindex += 1


    fig.show()








