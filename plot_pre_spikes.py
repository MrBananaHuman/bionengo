import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from nengo.utils.matplotlib import rasterplot

os.chdir('/home/pduggins/bionengo/data/S91GTG7QS/ens_bio2/')
signals_spikes=np.load('signals_spikes_from_ens_bio_to_ens_bio2.npz')
pre_spikes=signals_spikes['spikes_in']

sns.set(context='poster')
figure, ((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(4,2,sharex=True)
rasterplot(0.001*np.arange(0,pre_spikes.shape[0],1),pre_spikes,ax=ax1,use_eventplot=True)
ax1.set(ylabel='ens_bio ideal spikes',yticks=([]))
figure.savefig('ens_bio_ideal_spikes.png')