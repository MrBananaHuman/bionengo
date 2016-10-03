"""
Eric Hunsberger's Code for his 2016 Tech Report
Functions for estimating firing rate from a spike train
"""

import numpy as np

import scipy as sp
import scipy.interpolate
import scipy.stats

# def rate

def get_spike_times(t, spikes):
    if len(spikes) == len(t):
        return t[spikes.nonzero()]
    else:
        ### we were given spike times
        assert len(spikes) < len(t)
        return spikes

def p_kernel(t, spikes, kind='expon'):
    if kind == 'expon':
        kern = sp.stats.expon.pdf(t)
    elif kind == 'gauss':
        kern = sp.stats.normal.pdf(t)
    elif kind == 'alpha':
        avg_rate = spikes.sum() / len(t)
        kstd = 2. / avg_rate
        alpha = 1. / kstd     
        kern = alpha**2 * t * np.exp(-alpha*t)
    kern /= kern.sum()
    rates = np.array([np.convolve(kern, s, mode='full')[:len(t)] for s in spikes])
    return rates, kern

def kernel(t, spikes, kind='expon'):

    nt = len(t)
    dt = t[1] - t[0]
    assert spikes.shape[-1] == nt

    shape = spikes.shape
    if spikes.ndim > 2:
        spikes = spikes.reshape((-1,nt))
    elif spikes.ndim == 1:
        spikes = spikes.reshape((1, -1))

    n = spikes.shape[0]

    # max_rate = params['max_rate'][1]
    avg_rate = spikes.sum() / (n * t[-1])
    kstd = 2. / avg_rate
    # print avg_rate

    # kstd1 = 1./(max_rate/3)
    # kstd1 = 1./(max_rate*2)
    # kstd1 = 1./(avg_rate*2)

    kind = kind.lower()
    if kind == 'expon':
        kern = sp.stats.expon.pdf(np.arange(0, 5*kstd, dt), scale=kstd)
        # kern /= kern.sum()
        rates = np.array([np.convolve(kern, s, mode='full')[:nt] for s in spikes])
    elif kind == 'gauss':
        kern = sp.stats.normal.pdf(np.arange(-4*kstd, 4*kstd, dt), scale=kstd)
        # kern /= kern.sum()
        rates = np.array([np.convolve(kern, s, mode='same') for s in spikes])
    elif kind == 'expogauss':
        kern = sp.stats.expon.pdf(np.arange(0, 5*kstd, dt), scale=kstd)
        # kern /= kern.sum()
        rates = np.array([np.convolve(kern, s, mode='full')[:nt] for s in spikes])

        kstd2 = 1./(avg_rate*2)
        kern2 = sp.stats.norm.pdf(arange(-3*kstd2,3*kstd2,dt), scale=kstd2)
        kern2 /= kern2.sum()
        for i in xrange(len(rates)):
            rates[i] = np.convolve(kern2, rates[i], mode='same')
    elif kind == 'alpha':
        alpha = 1. / kstd
        tk = np.arange(0, 5*kstd, dt)
        kern = alpha**2 * tk * np.exp(-alpha*tk)
        # kern /= kern.sum()
        rates = np.array([np.convolve(s, kern, mode='full')[:nt] for s in spikes])

    return rates.reshape(shape), kern

def isi_hold_function(t, spikes, midpoint=False, interp='zero'):
    """Estimate firing rate using ISIs, with zero-order interpolation

    t : the times at which raw spike data (spikes) is defined
    spikes : the raw spike data
    midpoint : place interpolation points at midpoint of ISIs. Otherwise,
        the points are placed at the beginning of ISIs
    """

    spike_times = get_spike_times(t, spikes)
    isis = np.diff(spike_times)

    if midpoint:
        rt = np.zeros(len(isis)+2)
        rt[0] = t[0]
        rt[1:-1] = 0.5*(spike_times[0:-1] + spike_times[1:])
        rt[-1] = t[-1]

        r = np.zeros_like(rt)
        r[1:-1] = 1. / isis
    else:
        rt = np.zeros(len(spike_times)+2)
        rt[0] = t[0]
        rt[1:-1] = spike_times
        rt[-1] = t[-1]

        r = np.zeros_like(rt)
        r[1:-2] = 1. / isis

    # f = sp.interpolate.interp1d(rt, r, kind='zero',
    #                             copy=False, bounds_error=False, fill_value=0)
    return sp.interpolate.interp1d(rt, r, kind=interp, copy=False)

def isi_hold(t, spikes, **kwargs):
    f = isi_hold_function(t, spikes, **kwargs)
    return f(t)

def isi_smooth(t, spikes, width=0.05, **kwargs):
    f = isi_hold_function(t, spikes, **kwargs)
    rates = f(t)

    dt = t[1] - t[0]
    kern = sp.stats.norm.pdf(np.arange(-4*width, 4*width, dt), scale=width)
    kern /= kern.sum()
    rates = np.convolve(rates, kern, mode='same')
    # for i in xrange(len(rates)):
        # rates[i] = np.convolve(rates[i], kern, mode='same')

    return rates, kern

def adaptive_kernel(t, spikes):

    spike_times = get_spike_times(t, spikes)

    ### get rate estimate
    f_est = isi_hold_function(t, spikes)
    rate_ests = f_est(spike_times)

    rate = np.zeros_like(t)

    for st, r in zip(spike_times, rate_ests):
        # width = 1. / (1.5*r)
        width = 1. / (r)
        width = np.minimum(width, 0.2)
        bound = 4*width
        tmask = (t > st - bound) & (t < st + bound)
        kern = sp.stats.norm.pdf(t[tmask], loc=st, scale=width)
        rate[tmask] += kern

    # for st, r in zip(spike_times, rate_ests):
    #     # width = 1. / (1.5*r)
    #     width = 1. / (r)
    #     bound = 5*width
    #     tmask = (t > st) & (t < st + bound)
    #     kern = sp.stats.expon.pdf(t[tmask], loc=st, scale=width)
    #     rate[tmask] += kern

    # width = 0.02
    # kern = sp.stats.norm.pdf(np.arange(-4*width, 4*width, t[1] - t[0]), scale=width)
    # kern /= kern.sum()
    # rate = np.convolve(rate, kern, mode='same')

    # for st, r in zip(spike_times, rate_ests):
    #     # width = 1. / (1.5*r)
    #     width = 1. / (0.5*r)
    #     bound = 7*width
    #     tmask = (t >= st - width) & (t < st + bound)
    #     rv = sp.stats.gamma(2, loc=st-width, scale=width)
    #     kern = rv.pdf(t[tmask])

    #     # import matplotlib.pyplot as plt
    #     # plt.figure(99)
    #     # plt.clf()
    #     # plt.plot(t[tmask] - st, kern)

    #     # assert 0

    #     if np.isnan(kern).any():
    #         kern[np.isnan(kern)] = 0
    #         print "nans"

    #     # kern = sp.stats.gamma.pdf(t[tmask], a=2, loc=st, scale=width)
    #     rate[tmask] += kern


    return rate, kern
