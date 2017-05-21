from __future__ import print_function, division

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def read_header(fname):

    """
    :param fname:
    :return:

    header_length, bands, res_x, res_y, fmt
    """

    # grab header info
    header_length = open(fname, encoding='ISO-8859-1').read().find('\n.')
    header = open(fname, encoding='ISO-8859-1').read()[:header_length].split()
    bands = int(header[1])
    res_x = int(header[2])
    res_y = int(header[3])
    fmt = int(header[4])

    return header_length, bands, res_x, res_y, fmt

def read_hips(fname):

    header_length, bands, res_x, res_y, fmt = read_header(fname)

    # extract image from hips
    hips = np.fromfile(fname, np.float32)
    hips_length = len(hips)
    img = hips[hips_length - (bands * res_x * res_y):].reshape(bands, res_x, res_y)
    img = np.rot90(img.T, 3)

    return img, bands, res_x, res_y, fmt


def hipstats(fname, unique=False):

    img, bands, res_x, res_y, fmt = read_hips(fname)
    out = [img.min(), img.max(), img.mean(), img.std()]
    if unique:
        out.append(np.unique(img))

    return out


def hips2img(fname, order=[0,1,2], stretch=True, imshow=True,
             imsave=False, ax=None):
    
    img, bands, res_x, res_y, fmt = read_hips(fname)
    
    # data to display
    if order is None:
        img = np.nansum(img, axis=2)
        img = np.expand_dims(img, 2)
        order = 0
        cmap = 'gray'
        bands = 1
    else:
        if len(order) == 1 or bands == 1:
            order = order[0]
            cmap = 'gray'
        else:
            cmap = 'spectral'

    #stretch
    for b in range(bands):
        arr_b = img[:, :, b]
        if stretch:
            arr_b = ((arr_b - np.percentile(arr_b, 2.5)) / np.percentile(arr_b, 97.5))
            arr_b[arr_b < 0] = 0
            arr_b[arr_b > 1] = 1
        img[:, :, b] = arr_b

    # display image
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(img[:, :, order], cmap=cmap, interpolation='none')
    ax.axis('off')
    
    # save image
    if imsave:
        plt.imsave(os.path.splitext(fname)[0] + '.png', img[:, :, order],
                   cmap=cmap)

    # plot image to screen
    if imshow:
        plt.show()

    return ax



def hips2ani(fname, imsave=None, vmin=0, vmax=0.1):
    
    img, bands, res_x, res_y, fmt = read_hips(fname)
    
    fig = plt.figure(figsize=(10, 10))
    plt.axis('off')

    ims = []
    for b in range(bands):      
        im = plt.imshow(img[:, :, b], cmap='gray', interpolation='none', animated=True, 
            vmin=vmin, vmax=vmax)
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    
    # save image
    if imsave is not None:
        ani.save(imsave)

    return ani


