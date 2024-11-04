import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../utils')
from protocols import load_params
plt.style.use('paper_export')

protocols = ['SameRewDist', 'DistributionalRL_6Odours', 'Bernoulli', 'SameRewVar', 'SameVarReviewerSkewness', 'SameVarMaxSkewness']
# protocols = ['SameVarMaxSkewness']
max_ntt = 6
# protocols = ['Bernoulli']
fig, axs = plt.subplots(len(protocols), max_ntt, figsize=(max_ntt*1.3, len(protocols)*1.3), squeeze=False)

for ip, protocol in enumerate(protocols):

    print(protocol)

    colors, protocol_info, periods, kwargs = load_params(protocol)
    n_trace_types = protocol_info['n_trace_types']
    trace_type_names = protocol_info['trace_type_names']
    po = np.argsort(protocol_info['mean'][:n_trace_types])

    rew_bins = np.arange(-.25, 7.5, .5)  #np.arange(-.5, 8) if protocol != 'SameVarReviewerSkewness' else np.arange(-.25, 7.5, .5)
    rew_bin_centers = (rew_bins[:-1] + rew_bins[1:]) / 2
    max_rew = 8
    nskip = 4  # 2 if protocol != 'SameVarReviewerSkewness' else 4

    for i in range(max_ntt):
        ax = axs[ip, i]

        if i >= n_trace_types:
            ax.remove()

        else:

            ax.set_title(trace_type_names[i], y=0.95)
            ax.set_xlim([-1, max_rew])
            ax.set_ylim([0, 1.1])
            ax.set_xticks(rew_bin_centers[::nskip])
            ax.set_yticks([])
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            # axs[i].set_xlabel('Rew. amount (' + r' $\mu$L' + ')')

            hgts, _ = np.histogram(protocol_info['dists'][po[i]], bins=rew_bins, density=True)

            if i == 0:
                ax.set_ylabel('Probability')
                ax.set_yticks(np.arange(0, 1.1))

            ax.bar(rew_bin_centers, hgts * np.mean(np.diff(rew_bin_centers)), fc=colors['colors'][po[i]])


plt.tight_layout()
# plt.subplots_adjust(top=0.88, bottom=.2, wspace=.05, hspace=1.25)
# plt.subplots_adjust(top=0.88, bottom=.2, wspace=.15, hspace=.75)
plt.savefig('/home/adam/Documents/dist-rl/docs/manuscript/fig_components/all_tasks.pdf')
plt.savefig('/home/adam/Documents/dist-rl/docs/manuscript/fig_components/all_tasks.png', bbox_inches='tight')
# plt.show()