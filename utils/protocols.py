import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns

def get_cs_info(protocol):

    if protocol == 'DistributionalRL_6Odours' or protocol == 'DiverseDists':
        colors = [[0., 0.45, 0.45, 1],
                  [1, 0.8, 0.7, 1],
                  [.9, 0.6, 0.4, 1],
                  [.9, 0.3, 0., 1],
                  [0.15, 0.6, 0.6, 1],
                  [0.4, 0.85, 0.85, 1],
                  '#b47249']
                  # [0.541, 0.169, 0.886, 0.5]]
        # colors = ['#cec8ef', '#ffa07a', '#ff4500', '#800000', '#9932cc', '#663399']
                  # '#8A2BE2']
        trace_type_names = ['CS1', 'CS2', 'CS3', 'CS4', 'CS5', 'CS6']  # trial types with a trace period
        variable_rew_css = [1, 2, 3]

    elif protocol == 'Bernoulli':
        colors = list(plt.cm.copper(np.linspace(0, 1, 5))) + [[0.541, 0.169, 0.886, 0.5]] #  ['#8A2BE2']
        trace_type_names = ['0%', '20%', '50%', '80%', '100%']  # trial types with a trace period
        variable_rew_css = [1, 2, 3]

    elif protocol == 'Shock6Odor':
        colors = [[0., 0.45, 0.45, 1],
                  [1, 0.8, 0.7, 1],
                  [.9, 0.6, 0.4, 1],
                  [.9, 0.3, 0., 1],
                  [0.15, 0.6, 0.6, 1],
                  [0.4, 0.85, 0.85, 1],
                  [1, 0, 0, 1],
                  '#b47249']
                  # [0.541, 0.169, 0.886, 0.5]]
                  # '#8A2BE2']
        trace_type_names = ['CS1', 'CS2', 'CS3', 'CS4', 'CS5', 'CS6', 'Shock']  # trial types with a trace period
        variable_rew_css = [1, 2, 3]

    elif protocol == 'ShockBernoulli':
        colors = list(plt.cm.copper(np.linspace(0, 1, 5))) + [[1, 0, 0, 1], [0.541, 0.169, 0.886, 0.5]] #  ['#8A2BE2']
        trace_type_names = ['0%', '20%', '50%', '80%', '100%', 'Shock']  # trial types with a trace period
        variable_rew_css = [1, 2, 3]

    elif protocol == 'SameRewDist':
        # colors = ['#2832C2', '#287FC2', '#DF1FDF', '#FD92FD', '#043927', '#00A868', [0.541, 0.169, 0.886, 0.5]]
        # colors = ['#2832C2', '#287FC2', '#DF1FDF', '#FD92FD', '#B42020', '#FF5D5D', [0.541, 0.169, 0.886, 0.5]]
        colors = ['#9467bd', '#c5b0d5', '#d62728', '#ff9896', '#1f77b4', '#aec7e8', '#d28555']  #'#8c564b']
        trace_type_names = ['Nothing 1','Nothing 2','Fixed 1','Fixed 2','Variable 1', 'Variable 2']
        variable_rew_css = [4, 5]
    
    elif protocol == 'StimGradient':
        colors = ['#9467bd', '#d62728', '#1f77b4']
        trace_type_names = ['Nothing', 'Fixed', 'Variable']
        variable_rew_css = [2]

    elif protocol == 'TestStim':
        colors = ['b', 'r']
        trace_type_names = ['Rewarded', 'Stim']
        variable_rew_css = []

    elif protocol == 'SameRewSize':
        # colors = [print(mpl.colors.to_hex(cmap(x))) for x in [.01, .1, .45, .55, .9, .99]]
        colors = [plt.cm.spring(x) for x in [.01, .1, .45, .55, .9, .99]]
        trace_type_names = ['Nothing 1','Nothing 2','Small 1','Small 2','Big 1', 'Big 2']
        variable_rew_css = []

    elif protocol == 'SameRewVar':
        colors = ['#9467bd', '#c5b0d5', '#bb4513', '#b8860b', '#1f77b4', '#aec7e8', '#d28555']  #'#8c564b']
        trace_type_names = ['Nothing 1','Nothing 2','Uniform 1','Uniform 2','Bimodal 1', 'Bimodal 2']
        variable_rew_css = [4, 5]

    elif 'Skewness' in protocol:
        colors = ['#9467bd', '#c5b0d5', '#d62728', '#ff9896', '#17A589', '#48C9B0', '#d28555']  #'#8c564b']
        trace_type_names = ['Nothing 1','Nothing 2','Fixed 1','Fixed 2','Skewed 1', 'Skewed 2']
        variable_rew_css = [4, 5]

    else:
        raise Exception('Protocol type not recognized.')

    if protocol in ['StimGradient', 'TestStim', 'SameRewSize']:
        trial_type_names = trace_type_names
    else:
        trial_type_names = trace_type_names + ['Unexpected']

    vline_color = [0.25, 0.25, 0.25]
    n_trial_types = len(trial_type_names)
    n_trace_types = len(trace_type_names)
    return colors, vline_color, trial_type_names, trace_type_names, n_trial_types, n_trace_types, variable_rew_css

def load_params(protocol):
    # protocol-specific info
    colors, vline_color, trial_type_names, trace_type_names, n_trial_types, n_trace_types, _ = get_cs_info(protocol)

    if protocol == 'DistributionalRL_6Odours' or protocol == 'DiverseDists':
        exclude_tt = np.array([6])
        cs1_dist = np.zeros(50)
        cs2_dist = np.concatenate([2 * np.ones((8,)), 4 * np.ones((64,)), 6 * np.ones((8,))])
        cs3_dist = np.concatenate([2 * np.ones((35,)), 6 * np.ones((35,))])
        cs4_dist = np.concatenate(
            [np.ones((11,)), 2 * np.ones((11,)), 3 * np.ones((11,)), 4 * np.ones((11,)), 5 * np.ones((11,)),
             6 * np.ones((11,)), 7 * np.ones((11,))])
        cs5_dist = 2 * np.ones(25)
        cs6_dist = 6 * np.ones(25)
        dists = [cs1_dist, cs2_dist, cs3_dist, cs4_dist, cs5_dist, cs6_dist, np.array([4])]  # include unexpected reward
        null_tt = np.array([0])
        low_tt = 4
        high_tt = np.array([5])
        pairs_to_check = [(0., 5.)]  # for tukeyHSD result
        # prot_color = '#800000'
        prot_color = '#228B22'
        kwargs= {'phase': 7, 'same_avg_rew': [1, 2, 3], 'exclude_names': ('D1-11', 'D1-13')}

    elif protocol == 'Bernoulli':
        exclude_tt = np.array([5])
        rew_size = 6
        dist_0 = np.zeros(50)
        dist_20 = np.concatenate([np.zeros(80), rew_size * np.ones(20)])
        dist_50 = np.concatenate([np.zeros(50), rew_size * np.ones(50)])
        dist_80 = np.concatenate([np.zeros(20), rew_size * np.ones(80)])
        dist_100 = rew_size * np.ones(50)
        dists = [dist_0, dist_20, dist_50, dist_80, dist_100, np.array([rew_size])] # include unexpected reward
        null_tt = np.array([0])
        low_tt = 0
        high_tt = np.array([4])
        pairs_to_check = [(0., 4.)]  # for tukeyHSD result
        prot_color = '#CC5500'
        kwargs = {'phase': 4, 'same_avg_rew': [], 'exclude_sess': (("AL28", 20210415),)}

    elif protocol == 'Shock6Odor':
        exclude_tt = np.array([7])  # TODO: should I also exclude shock CS (6)?
        cs1_dist = np.zeros(25)
        cs2_dist = np.concatenate([2 * np.ones((4,)), 4 * np.ones((32,)), 6 * np.ones((4,))])
        cs3_dist = np.concatenate([2 * np.ones((20,)), 6 * np.ones((20,))])
        cs4_dist = np.concatenate(
            [np.ones((4,)), 2 * np.ones((4,)), 3 * np.ones((4,)), 4 * np.ones((4,)), 5 * np.ones((4,)),
             6 * np.ones((4,)), 7 * np.ones((4,))])
        cs5_dist = 2 * np.ones(25)
        cs6_dist = 6 * np.ones(25)
        cs7_dist = np.zeros(100)
        dists = [cs1_dist, cs2_dist, cs3_dist, cs4_dist, cs5_dist, cs6_dist, cs7_dist, np.array([4])]
        null_tt = np.array([0])
        low_tt = 4
        high_tt = np.array([5])
        prot_color = '#000000'
        pairs_to_check = [(0., 5.)]  # for tukeyHSD result

        kwargs = {'phase': 7, 'same_avg_rew': [1, 2, 3]}

    elif protocol == 'ShockBernoulli':
        exclude_tt = np.array([6])  # TODO: should I also exclude shock CS (5)?
        rew_size = 6
        dist_0 = np.zeros(50)
        dist_20 = np.concatenate([np.zeros(40), rew_size * np.ones(10)])
        dist_50 = np.concatenate([np.zeros(25), rew_size * np.ones(25)])
        dist_80 = np.concatenate([np.zeros(10), rew_size * np.ones(40)])
        dist_100 = rew_size * np.ones(50)
        dist_shock = np.zeros(100)
        dists = [dist_0, dist_20, dist_50, dist_80, dist_100, dist_shock, np.array([rew_size])]
        null_tt = np.array([0])
        low_tt = 0
        high_tt = np.array([4])
        prot_color = '#000000'
        pairs_to_check = [(0., 4.)]  # for tukeyHSD result
        kwargs = {'phase': 4, 'same_avg_rew': []}

    elif protocol == 'SameRewDist':
        exclude_tt = np.array([6])
        csa1_dist = np.zeros(40)
        csa2_dist = np.zeros(40)
        csb1_dist = 4*np.ones(40)
        csb2_dist = 4*np.ones(40)
        csc1_dist = np.concatenate([2*np.ones(35), 6*np.ones(35)])
        csc2_dist = np.concatenate([2*np.ones(35), 6*np.ones(35)])
        dists = [csa1_dist, csa2_dist, csb1_dist, csb2_dist, csc1_dist, csc2_dist, np.array([4])]
        null_tt = np.array([0, 1])
        low_tt = 0
        high_tt = np.array([2, 3, 4, 5])
        pairs_to_check = [(0., 2.), (0., 3.), (1., 2.), (1., 3.)]  # for tukeyHSD result
        prot_color = '#00CED1'
        kwargs = {'phase': 3, 'same_avg_rew': [2, 3, 4, 5], 
                  'id_mean_swap_inds': [(np.array([0, 2, 3]), np.array([1, 4, 5])),
                                        (np.array([0, 2, 3]), np.array([1, 5, 4]))],
                  # given to construct_where_str. These are lesion animals for which I missed the
                  # lesion, was not able to confirm the lesion, or where I was only
                  # able to record the non-lesioned side. Include for control dataset and exclude
                  # from lesion dataset. Needs to be of length > 1, which is why AL00 is there as a dummy
                  'additional_names': ("AL77", "AL81", "AL79", "AL82")
                  # 'additional_names': ("AL77", "AL81")
                  }

    elif protocol == 'StimGradient':
        exclude_tt = np.array([99])  # no exclude trials, but need to give it something
        nothing_dist = np.zeros(40)
        fixed_dist = 4*np.ones(40)
        variable_dist = np.concatenate([np.zeros(20), 8*np.ones(20)])
        dists = [nothing_dist, fixed_dist, variable_dist]
        null_tt = np.array([0])
        low_tt = 0
        high_tt = np.array([1, 2])
        pairs_to_check = [(0., 1.),(0., 2.)]
        prot_color = '#008080'
        kwargs = {'phase': 1, 'same_avg_rew': [1, 2]}

    elif protocol == 'TestStim':
        exclude_tt = np.array([99])  # no exclude trials, but need to give it something
        cs1_dist = 6*np.ones(40)
        cs2_dist = np.zeros(40)
        dists = [cs1_dist, cs2_dist]
        null_tt = np.array([1])
        low_tt = 1
        high_tt = np.array([0])
        pairs_to_check = [(0., 1.)]
        prot_color = '#710C04'
        kwargs = {'phase': 1, 'same_avg_rew': []}

    elif protocol == 'SameRewSize':
        exclude_tt = np.array([99])  # no exclude trials, but need to give it something
        nothing_dist = np.zeros(40)
        small_dist = 3*np.ones(40)
        big_dist = 6*np.ones(40)
        dists = [np.copy(nothing_dist), np.copy(nothing_dist), np.copy(small_dist),
                 np.copy(small_dist), np.copy(big_dist), np.copy(big_dist)]
        null_tt = np.array([0, 1])
        low_tt = 0
        high_tt = np.array([4, 5])
        pairs_to_check = [(0., 4.),(0., 5.), (1., 4.),(1., 5.)]
        prot_color = '#805a17'
        kwargs = {'phase': 1, 'same_avg_rew': []}

    elif protocol == 'SameRewVar':
        exclude_tt = np.array([6])
        csa1_dist = np.zeros(39)
        csa2_dist = np.zeros(39)
        csb1_dist = np.repeat(np.arange(1, 8), 16)
        csb2_dist = np.repeat(np.arange(1, 8), 16)
        csc1_dist = np.concatenate([2*np.ones(28), 6*np.ones(28)])
        csc2_dist = np.concatenate([2*np.ones(28), 6*np.ones(28)])
        dists = [csa1_dist, csa2_dist, csb1_dist, csb2_dist, csc1_dist, csc2_dist, np.array([4])]
        null_tt = np.array([0, 1])
        low_tt = 0
        high_tt = np.array([2, 3, 4, 5])
        pairs_to_check = [(0., 2.), (0., 3.), (1., 2.), (1., 3.)]  # for tukeyHSD result
        prot_color = '#BB4513'
        kwargs = {'phase': 3, 'same_avg_rew': [2, 3, 4, 5], 
                  'id_mean_swap_inds': [(np.array([0, 2, 3]), np.array([1, 4, 5])),
                                        (np.array([0, 2, 3]), np.array([1, 5, 4]))],
                  'exclude_names': ('AL100', '')
                  }

    elif protocol == 'SameVarReviewerSkewness':
        exclude_tt = np.array([6])
        nothing_dist1 = np.zeros(100)
        nothing_dist2 = np.zeros(100)
        fixed_dist1 = 4.5*np.ones(100)
        fixed_dist2 = 4.5*np.ones(100)
        skewed_dist1 = np.concatenate([np.zeros(5), 2*np.ones(15), 4*np.ones(30), 6*np.ones(50)])
        skewed_dist2 = np.concatenate([np.zeros(5), 2*np.ones(15), 4*np.ones(30), 6*np.ones(50)])
        dists = [nothing_dist1, nothing_dist2, fixed_dist1, fixed_dist2, skewed_dist1, skewed_dist2]
        null_tt = np.array([0, 1])
        low_tt = 0
        high_tt = np.array([2, 3, 4, 5])
        pairs_to_check = [(0., 2.), (0., 3.), (1., 2.), (1., 3.)]  # for tukeyHSD result
        prot_color = '#BB4513'
        kwargs = {'phase': 3, 'same_avg_rew': [2, 3, 4, 5], 
              'id_mean_swap_inds': [(np.array([0, 2, 3]), np.array([1, 4, 5])),
                                    (np.array([0, 2, 3]), np.array([1, 5, 4]))],
              }

    elif protocol == 'SameVarMaxSkewness':
        exclude_tt = np.array([6])
        nothing_dist1 = np.zeros(100)
        nothing_dist2 = np.zeros(100)
        fixed_dist1 = 4*np.ones(100)
        fixed_dist2 = 4*np.ones(100)
        # skewed_dist1 = np.concatenate([3*np.ones(80), 8*np.ones(20)])
        # skewed_dist2 = np.concatenate([3*np.ones(80), 8*np.ones(20)])
        skewed_dist1 = np.concatenate([np.zeros(20), 5 * np.ones(80)])
        skewed_dist2 = np.concatenate([np.zeros(20), 5 * np.ones(80)])
        dists = [nothing_dist1, nothing_dist2, fixed_dist1, fixed_dist2, skewed_dist1, skewed_dist2]
        null_tt = np.array([0, 1])
        low_tt = 0
        high_tt = np.array([2, 3, 4, 5])
        pairs_to_check = [(0., 2.), (0., 3.), (1., 2.), (1., 3.)]  # for tukeyHSD result
        prot_color = '#BB4513'
        kwargs = {'phase': 3, 'same_avg_rew': [2, 3, 4, 5],
              'id_mean_swap_inds': [(np.array([0, 2, 3]), np.array([1, 4, 5])),
                                    (np.array([0, 2, 3]), np.array([1, 5, 4]))],
              }

    else:
        raise Exception('Protocol not recognized')

    # kwargs that I want to be consistent across protocols
    kwargs['n_trial'] = 150
    kwargs['quality'] = 2
    kwargs['curated'] = 1
    # kwargs['continuous'] = 1  # ignored if ephys
    kwargs['significance'] = 1  # now computed based off of Mann-Whitney U between null and high trial type
    kwargs['stats'] = 'NULL'  # or 'p_kruskal' or 'p_anova' or 'tukeyHSD'
    kwargs['probe1_region'] = 'striatum'  # ignored if imaging
    kwargs['wavelength'] = 1000
    kwargs['code'] = (3, 4, 5)
    kwargs['manipulation'] = None

    # colors = [c for i, c in enumerate(colors) if i not in exclude_tt]
    palette = sns.color_palette(colors)
    lw = 2  # line width for plotting lines
    colors = {'colors': np.array([mpl.colors.to_hex(x) for x in colors], dtype='object'),
              'palette': palette,
              'vline_color': vline_color,
              'lw': lw,
              'prot_color': prot_color
              }

    norm_factor = max([d for subl in dists for d in subl])
    var_types = [i for i, x in enumerate(dists) if len(np.unique(x)) > 1]  # [1, 2, 3] for 6 Odor and Bernoulli

    id_dists = np.zeros((n_trace_types, n_trace_types), dtype=bool)
    for i in range(n_trace_types):
        for j in range(n_trace_types):
            id_dists[i, j] = np.array_equal(dists[i], dists[j])

    protocol_info = {'protocol': protocol,
                     'trace_type_names': trace_type_names,
                     'trial_type_names': trial_type_names,
                     'n_trace_types': n_trace_types,
                     'n_trial_types': n_trial_types,
                     # make unexpected reward have an "expected" mean/variance of nan. Just need to tack on something for resid_mean and resid_var
                     'mean': [np.mean(x) for x in dists[:n_trace_types]] + [np.nan],
                     'resid_mean': [np.mean(x) for x in dists[:n_trace_types]] + [np.nan],
                     'var': [np.var(x) for x in dists[:n_trace_types]] + [np.nan],
                     'resid_var': [np.var(x) for x in dists[:n_trace_types]] + [np.nan],
                     'cvar': [np.mean(x[x <= np.quantile(x, .1)]) for x in dists[:n_trace_types]] + [np.nan],
                     'resid_cvar': [np.mean(x[x <= np.quantile(x, .1)]) for x in dists[:n_trace_types]] + [np.nan],
                     'std': [np.std(x) for x in dists[:n_trace_types]],
                     'dists': dists,
                     'norm_dists': [subl / norm_factor for subl in dists],
                     'n_dists': len(dists),
                     'var_types': var_types,
                     'exclude_tt': exclude_tt,
                     'null_tt': null_tt,
                     'low_tt': low_tt,
                     'high_tt': high_tt,
                     'pairs_to_check': pairs_to_check,
                     'id_dist_inds': np.nonzero(np.triu(id_dists, k=1)),
                     'exclude_shift': 3}  # 3 seconds is the difference between expected and unexpected (1 sec odor + 2 secs trace)

    period_names = ['Baseline', 'Odor', 'Early Trace', 'Late Trace', 'Reward']
    period_abbr = ['base', 'odor', 'et', 'lt', 'rew']
    periods_to_plot = np.array([1, 2, 3])  # odor, early and late trace
    periods = {'n_periods': len(period_names),
               'n_comp_periods': len(period_names) - 1,
               'n_prerew_periods': len(period_names) - 1,
               'period_names': period_names,
               'period_abbr': period_abbr,
               'periods_to_plot': periods_to_plot,
               'n_periods_to_plot': len(periods_to_plot),
               'var_types': var_types,
               'alpha': 0.05}  # for significance testing

    return colors, protocol_info, periods, kwargs