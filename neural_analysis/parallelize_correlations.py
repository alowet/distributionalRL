import numpy as np
from analysisUtils import find_trial_rew_correls

def parallelize_correlations(start_cell, end_cell, inds, trial_types, binned, trace_inds, trace_mean, corrs,
                             trace_dur_corrs, prerew_keys, postrew_keys, protocol_info, rewards, cue_resps,
                             cue_spk_cnts, all_scrambled_order, scrambled_cue_resps, scrambled_cue_spk_cnts, n_splits):

    rng = np.random.default_rng()
    cells_this_sess = end_cell - start_cell
    n_trace_types = protocol_info['n_trace_types']

    loc_corrs = corrs
    loc_trace_dur_corrs = trace_dur_corrs

    # correls without train/test split
    for key in prerew_keys + postrew_keys:
        if key in prerew_keys:
            full_stat = np.array(protocol_info[key])[trial_types[trace_inds]]
            loc_trace_dur_corrs[key][start_cell:end_cell, :] = find_trial_rew_correls(trace_mean, full_stat)[:, 0, :]
        elif key == 'rew':
            full_stat = rewards[trace_inds]
        elif key == 'rpe':
            full_stat = rewards[trace_inds] - np.array(protocol_info['mean'])[trial_types[trace_inds]]
        loc_corrs[key]['ord']['all'].dat[start_cell:end_cell, :, :] = find_trial_rew_correls(binned[:, trace_inds, :], full_stat)

    scrambled_order = np.array([rng.permutation(x) for x in
                                np.tile(np.arange(n_trace_types)[np.newaxis, :], [cells_this_sess, 1])]).T
    all_scrambled_order[:, start_cell:end_cell] = scrambled_order
    scrambled_cue_resps[:, start_cell:end_cell] = np.take_along_axis(cue_resps[:, start_cell:end_cell],
                                                                     scrambled_order[..., np.newaxis, np.newaxis],
                                                                     axis=0)
    scrambled_cue_spk_cnts[:, start_cell:end_cell] = np.take_along_axis(cue_spk_cnts[:, start_cell:end_cell],
                                                                        scrambled_order[
                                                                            ..., np.newaxis, np.newaxis], axis=0)
    # shuffle odor identities to see what comes out
    scrambled_inds = np.array(inds, dtype='object')[scrambled_order]

    for i_cell in range(cells_this_sess):

        all_scrambled_activity = np.concatenate([binned[i_cell, scrambled_inds[i_type, i_cell], :] for i_type in range(n_trace_types)], axis=0)
        n_trials_per_scrambled_type = [len(scrambled_inds[i_type, i_cell]) for i_type in range(n_trace_types)]

        for key in prerew_keys + postrew_keys:

            if key in prerew_keys:
                scrambled_stat_vec = np.concatenate([[protocol_info[key][i]] * x for i, x in enumerate(n_trials_per_scrambled_type)])
            elif key == 'rew':
                scram_rew_vec = np.concatenate(
                    [rewards[scrambled_inds[i_type, i_cell]] for i_type in range(n_trace_types)])
                scrambled_stat_vec = scram_rew_vec
            elif key == 'rpe':
                scram_mean_vec = np.concatenate([[protocol_info['mean'][i]] * x for i, x in enumerate(n_trials_per_scrambled_type)])
                scrambled_stat_vec = scram_rew_vec - scram_mean_vec
            loc_corrs[key]['scram']['all'].dat[start_cell + i_cell, :, :] = find_trial_rew_correls(
                all_scrambled_activity[np.newaxis, ...], scrambled_stat_vec)

            for i_split in range(n_splits):

                scrambled_activity = np.concatenate(
                    [np.array(binned[i_cell, scrambled_inds[i_type, i_cell][i_split::n_splits], :]) for i_type in
                     range(n_trace_types)], axis=0)
                n_split_trials_per_scrambled_type = [len(scrambled_inds[i_type, i_cell][i_split::n_splits])
                                                     for i_type in range(n_trace_types)]

                if key in prerew_keys:
                    split_scrambled_stat_vec = np.concatenate(
                        [[protocol_info[key][i]] * x for i, x in enumerate(n_split_trials_per_scrambled_type)])
                elif key == 'rew':
                    split_scrambled_stat_vec = np.concatenate(
                        [rewards[scrambled_inds[i_type, i_cell][i_split::n_splits]] for i_type in range(n_trace_types)])
                elif key == 'rpe':
                    # need to repeat this because of the splits
                    split_scram_rew_vec = np.concatenate(
                        [rewards[scrambled_inds[i_type, i_cell][i_split::n_splits]] for i_type in
                         range(n_trace_types)])
                    split_scram_mean_vec = np.concatenate(
                        [[protocol_info['mean'][i]] * x for i, x in enumerate(n_split_trials_per_scrambled_type)])
                    split_scrambled_stat_vec = split_scram_rew_vec - split_scram_mean_vec

                loc_corrs[key]['scram']['split'].dat[start_cell + i_cell, :, :, i_split] = \
                    find_trial_rew_correls(scrambled_activity[np.newaxis, ...], split_scrambled_stat_vec)

                if i_cell == 0:

                    activity = np.concatenate(
                        [np.array(binned[:, inds[i_type][i_split::n_splits], :]) for i_type in range(n_trace_types)],
                        axis=1)
                    n_trials_per_type = [len(inds[i_type][i_split::n_splits]) for i_type in range(n_trace_types)]

                    if key in prerew_keys:
                        stat_vec = np.concatenate(
                            [[protocol_info[key][i]] * x for i, x in enumerate(n_trials_per_type)])
                    elif key == 'rew':
                        # trialwise correlation of actual reward delivered on a given trial, excluding unexpected reward for consistency
                        stat_vec = np.concatenate(
                            [rewards[inds[i_type][i_split::n_splits]] for i_type in range(n_trace_types)])
                    elif key == 'rpe':
                        # need to repeat this because of the splits
                        rew_vec = np.concatenate(
                            [rewards[inds[i_type][i_split::n_splits]] for i_type in
                             range(n_trace_types)])
                        # trialwise correlation of RPE (actual minus expected reward)
                        mean_vec = np.concatenate(
                            [[protocol_info['mean'][i]] * x for i, x in enumerate(n_trials_per_type)])
                        stat_vec = rew_vec - mean_vec

                    loc_corrs[key]['ord']['split'].dat[start_cell:end_cell, :, :, i_split] = \
                        find_trial_rew_correls(activity, stat_vec)

    corrs = loc_corrs
    trace_dur_corrs = loc_trace_dur_corrs

