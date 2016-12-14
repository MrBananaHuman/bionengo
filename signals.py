'''Eric Hunsberger's Code for his 2016 Tech Report'''

import numpy as np
import numpy.random as npr

import scipy as sp
import scipy.interpolate
import scipy.stats

def constant(dt,t_final,value):
    return np.ones(int(t_final/dt))*1.0*value

def prime_sinusoids(dt,t_final,dim):
    import ipdb
    def checkprime(n):
        d=2
        while d<=(n/2):
            if n%d==0:
                return 0
            d+=1
        return 1

    def primeno(term):
        numbers=[]
        i=2
        while len(numbers)<term:
            if checkprime(i)==1:
                numbers.append(i)
            i+=1
        return numbers

    def sinusoids(t_final,primes,t):
        import numpy as np
        return np.array([np.sin(t*2*np.pi*hz) for hz in primes])

    primes=primeno(dim)
    timesteps=np.arange(0,t_final,dt)
    output=sinusoids(t_final,primes,timesteps)
    return output

def equalpower(dt, t_final, max_freq, mean=0.0, std=1.0, n=None):
    """Generate a random signal with equal power below a maximum frequency

    Parameters
    ----------
    dt : float
        Time difference between consecutive signal points [in seconds]
    t_final : float
        Length of the signal [in seconds]
    max_freq : float
        Maximum frequency of the signal [in Hertz]
    mean : float
        Signal mean (default = 0.0)
    std : float
        Signal standard deviation (default = 1.0)
    n : integer
        Number of signals to generate

    Returns
    -------
    s : array_like
        Generated signal(s), where each row is a time, and each column a signal
    """

    vector_out = n is None
    n = 1 if n is None else n

    df = 1. / t_final    # fundamental frequency

    nt = np.round(t_final / dt)        # number of time points / frequencies
    nf = np.round(max_freq / df)      # number of non-zero frequencies
    assert nf < nt

    theta = 2*np.pi*npr.rand(n, nf)
    B = np.cos(theta) + 1.0j * np.sin(theta)

    A = np.zeros((n,nt), dtype=np.complex)
    A[:,1:nf+1] = B
    A[:,-nf:] = np.conj(B)[:,::-1]

    S = np.fft.ifft(A, axis=1).real

    S = (std / S.std(axis=1))[:,None] * (S - S.mean(axis=1)[:,None] + mean)
    if vector_out: return S.flatten()
    else:          return S


def poisson_binary(dt, t_final, mean_freq, max_freq, low=-1.0, high=1.0, n=None):
    """Generate a random signal that switches between two values"""

    assert max_freq < 0.5/dt
    assert mean_freq < max_freq
    exp = sp.stats.expon(loc=(1./max_freq), scale=(1./mean_freq - 1./max_freq))

    vector_out = n is None
    n = 1 if n is None else n
    nt = np.round(t_final / dt)
    s = np.zeros((n, nt))

    for i in xrange(n):
        t = 0.0
        cur_low = True

        while t < t_final:
            tau = exp.rvs()      # draw random sample
            val = low if cur_low else high
            s[i, np.round(t/dt):np.round((t+tau)/dt)] = val
            t += tau
            cur_low = not cur_low

    if vector_out: return s.flatten()
    else:          return s


def white_binary(dt, t_final, mean=0.0, std=1.0, n=None):
    vector_out = n is None
    n = 1 if n is None else n
    nt = np.round(t_final / dt)
    # s = np.zeros((n, nt))
    s = npr.rand(n, nt)
    s[:] = (s > 0.5)
    s[:] = (s - s.mean(1)[:,None]) / s.std(1)[:,None]

    if vector_out: return s.flatten()
    else:          return s


# def white_gaussian(dt, t_final, loc=0.0, scale=1.0, n=None):
#     vector_out = n is None
#     n = 1 if n is None else n
#     nt = np.round(t_final / dt)

def get_rng(kind, **params):
    kind = kind.lower()
    if kind in ['gaussian', 'gauss', 'normal']:
        return lambda size: npr.normal(size=size, **params)
    elif kind in ['uniform']:
        return lambda size: npr.uniform(size=size, **params)
    elif kind in ['binary', 'bernoulli']:
        low = params.get('low', -1)
        high = params.get('high', 1)
        return lambda size: (high - low)*npr.binomial(1, 0.5, size=size) + low
    else:
        raise ValueError('Unrecognized kind')


def white(dt, t_final, n=None, kind='gaussian', **params):
    vector_out = n is None
    n = 1 if n is None else n
    nt = np.round(t_final / dt)
    size = (n, nt)

    f = get_rng(kind, **params)
    s = f(size)

    if vector_out: return s.flatten()
    else:          return s


def switch(dt, t_final, max_freq, n=None, kind='gaussian', **params):
    """Generates sample-and-hold noise.

    Flat below a certain frequency, PSD rolls off with -10 dB / decade,
    but with notches at multiples of the switching frequency.
    (1/2 roll off of Ornstein-Uhlenbeck process)
    """

    vector_out = n is None
    n = 1 if n is None else n
    nt = np.round(t_final / dt)
    # size = (n, nt)

    # nf = int(max_freq/dt)
    df = 1. / max_freq
    nf = int(t_final*max_freq)
    size = (n, nf)

    t = np.linspace(dt, t_final, nt)
    t0 = np.linspace(0, t_final-df, nf)

    f = get_rng(kind, **params)
    S = f(size)

    f = sp.interpolate.interp1d(t0, S, kind='zero', axis=-1,
                                bounds_error=False, fill_value=S[:,0])
    s = f(t)

    if vector_out: return s.flatten()
    else:          return s


def poisson(dt, t_final, mean_freq, max_freq, n=None, kind='gaussian', **params):
    """Generate a random signal that switches between random values,
    where the switches are exponentially distributed.

    Flat below a certain frequency, PSD rolls off with -10 dB / decade
    (1/2 roll off of Ornstein-Uhlenbeck process).
    """

    assert max_freq < 0.5/dt
    assert mean_freq < max_freq
    exp = sp.stats.expon(loc=(1./max_freq), scale=(1./mean_freq - 1./max_freq))

    vector_out = n is None
    n = 1 if n is None else n
    nt = np.round(t_final / dt)
    s = np.zeros((n, nt))

    rng = get_rng(kind, **params)
    for i in xrange(n):
        t = 0.0
        cur_low = True

        while t < t_final:
            # draw random time sample
            tau = exp.rvs()
            s[i, np.round(t/dt):np.round((t+tau)/dt)] = rng(1)
            t += tau

    if vector_out: return s.flatten()
    else:          return s


def pink_noise(dt, t_final, mean=0.0, std=1.0, n=None):
    vector_out = (n is None)
    n = 1 if n is None else n

    df = 1. / t_final
    nt = int(np.round(t_final / dt))
    f = np.linspace(0, (nt-1)*df, nt)

    phase = npr.randn(n,nt) + 1.0j*npr.randn(n,nt)
    phase /= np.abs(phase)

    A = np.zeros((n,nt), dtype=np.complex)
    A[:,:] = np.sqrt(1. / f)[None,:] * phase
    A[:,0] = 0
    if nt % 2 == 0:
        A[:,nt/2+1:] = A[:,nt/2-1:0:-1].conj()
    else:
        A[:,nt/2+1:] = A[:,nt/2:0:-1].conj()

    S = np.fft.ifft(A, axis=1).real
    S = (std / S.std(axis=1))[:,None] * (S - S.mean(axis=1)[:,None] + mean)
    if vector_out: return S.flatten()
    else:          return S


def show_inputs():
    import matplotlib.pyplot as plt
    plt.ion()

    dt = 1e-3
    t_final = 4

    t = np.linspace(dt, t_final, t_final/dt)
    us = [
        equalpower(dt, t_final, max_freq=2.5),
        switch(dt, t_final, 5.0, kind='uniform', low=-1, high=1),
        poisson(dt, t_final, mean_freq=5.0, max_freq=10.0,
                kind='uniform', low=-1, high=1)
        ]
    names = ['equal power', 'regular switch', 'exponential switch']

    plt.figure(1)
    plt.clf()
    rows = 2
    cols = len(us)

    for i in xrange(len(us)):
        u, name = us[i], names[i]
        plt.subplot(rows, cols, i+1)
        plt.plot(t, u)
        plt.xlabel('time [s]')
        plt.title(name)

        plt.subplot(rows, cols, cols+i+1)
        fmax = 100
        f = np.linspace(0, 1./dt, len(u))
        F = np.fft.fft(u)
        plt.loglog(f[f < fmax], np.abs(F[f < fmax]))
        plt.xlabel('frequency [Hz]')

    plt.tight_layout()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    dt = 1e-3
    t_final = 10
    # n = None
    n = 100
    dt2 = np.sqrt(dt)

    t = np.linspace(dt, t_final, t_final/dt)
    # u = pink_noise(dt, t_final, n=n)
    # u = poisson_binary(dt, t_final, 1, 5, n=n)
    # u = white_binary(dt, t_final, n=n)

    # u = white(dt, t_final, n=n, kind='normal', scale=1/dt2)
    # u = white(dt, t_final, n=n, kind='uniform', low=-1/dt2, high=1/dt2)
    # u = white(dt, t_final, n=n, kind='binary', low=-1/dt2, high=1/dt2)

    # u = switch(dt, t_final, 5.0, n=n, kind='normal', scale=1)
    # u = switch(dt, t_final, 5.0, n=n, kind='uniform', low=-1, high=1)

    u = poisson(dt, t_final, mean_freq=5.0, max_freq=10.0, n=n,
    #             # kind='gaussian', scale=1.0)
                kind='uniform', low=-1, high=1)
    #             kind='binary', low=-1, high=1)

    # plt.figure(1)
    # plt.clf()
    # plt.plot(t, u[:10].T)

    plt.figure(2)
    plt.clf()
    df = 1. / t_final
    f = np.linspace(0, 1./dt - df, len(t))
    F = np.fft.fft(u, axis=1)
    f2 = len(f) / 2

    r, c = (1, 2)

    plt.subplot(r, c, 1)
    plt.loglog(f[1:f2], np.abs(F).mean(0)[1:f2])

    plt.subplot(r, c, 2)
    plt.semilogx(f[1:f2], np.angle(F).mean(0)[1:f2])

    # plt.subplot(r, c, 3)
    # plt.semilogx(f[1:f2], np.angle(F).std(0)[1:f2])

    # plt.subplot(r, c, 4)
    # plt.hist(np.angle(F).flatten(), bins=100)

    A = np.angle(F).mean(0)
    print A.std()

# def lowpass_filter_conv(S, dt, tau_c, S0=None):
#     S = np.matrix(S)
#     nt = S.shape[1]
#     if S0 is not None:
#         S[:,0] = S0
#     else:
#         ind = np.round(5*tau_c / dt)
#         S[:,0] = S[:,:min(ind, nt)].mean(axis=1)


# def lowpass_filter(S, dt, tau_c, S0=None):
#     S = np.matrix(S)
#     nt = S.shape[1]
#     if S0 is not None:
#         S[:,0] = S0
#     else:
#         ind = np.round(5*tau_c / dt)
#         S[:,0] = S[:,:min(ind, nt)].mean(axis=1)

#     ### filtering is much faster when done with a loop than with convolution
#     # alpha = dt / tau_c           # forward Euler
#     alpha = dt / (tau_c + dt)    # reverse Euler
#     for i in xrange(1,nt):
#         S[:,i] = S[:,i-1] + alpha*(S[:,i] - S[:,i-1])

#     S = np.asarray(S)
#     if S.shape[0] == 1: return S.flatten()
#     else:               return S
