include("decoding_utils.jl")

#################
# Hyperparameters
#################

time_window_idx = 4

# folder of the project
folder = joinpath(folder_path, "decoding_SameRewDist_pseudo_transfer")
if time_window_idx == 2
prefix = "decoding_SameRewDist_pseudo_transfer_0to1_"
elseif time_window_idx == 4
prefix = "decoding_SameRewDist_pseudo_transfer_2to3_"
end

datafile_same = joinpath(folder_path, "SameRewDist_combined_spks_data_20230918.p")
same_data = myunpickle(datafile_same)

num_networkstat = 16

standard_nsplits = 6

num_pseudotrial_per_dist = 200

task_names = ["SameRewDist"]

n_dists = 6
dist_rng = collect(1:n_dists)

n_transfer_dists = 4
n_test_dists = 2

standard_set = dist_rng
standard_subset = [1, 2, 3, 5]  # the ground-truth distributions that we will give to the decoder each time
standard_test = [4, 6]
# the data that we'll give to the decoder each time. The test data can be inferred from which distributions are absent 
matched_sets = [[1, 2, 3, 5], [1, 2, 3, 6], [1, 2, 4, 5], [1, 2, 4, 6]]  # e.g. for the first set, try testing with Fixed = 4 and Variable = 6
missed_sets = [[1, 2, 3, 4], [1, 2, 4, 3], [1, 2, 5, 6], [1, 2, 6, 5]]  # e.g. for the first set, try testing with Variable = 5 and Variable = 6
n_transfers = length(matched_sets) + length(missed_sets)  # = 8

generate = true
outer_train = true
network_train = true
do_collect = true

cutoffs = [73]
n_reward_values = 4

for d in [1]
    
    data, resp_data, session_id_data, mouse_ids, session_ids, distributions, reward_values, CDFs, CDFs_without_one, constant_intervals, 
        quantiles_on_constant_intervals, regions, dorso_ventral_coordinates = setup_decoding(same_data, d, time_window_idx, dist_rng)

    for cutoff in cutoffs

        cut_key = "cutoff_" * string(cutoff)
        dist_neuron_labels = data["neuron_info"][cut_key]
        
        println("Cutoff: ", cutoff)

        ########################
        # Generate pseudo trials
        ########################

        if generate

            selection_indices = [u for u in 1:length(dist_neuron_labels) if dist_neuron_labels[u]]

            session_neuron_counts = zeros(Int64,(length(session_ids)))
            for session_id_idx in 1:length(session_ids)
                session_id = session_ids[session_id_idx]
                data_session = data_for_selection_session(resp_data, selection_indices, session_id_data, session_id)
                if size(data_session)[2] == 0
                    continue
                end
                session_neuron_counts[session_id_idx] = size(data_session)[2]
            end
            println(session_neuron_counts,"  ", sum(session_neuron_counts))

            standard_data = [zeros((length(selection_indices), n_dists * num_pseudotrial_per_dist)) for i in 1:standard_nsplits]
            transfer_data = [zeros((length(selection_indices), n_transfer_dists * num_pseudotrial_per_dist)) for i in 1:n_transfers]
            test_data = [zeros((length(selection_indices), n_test_dists * num_pseudotrial_per_dist)) for i in 1:n_transfers]
            # test_data = zeros((n_transfers, length(selection_indices), 2 * num_pseudotrial_per_dist))

            standard_dist = zeros((size(distributions)[2], n_dists * num_pseudotrial_per_dist))
            transfer_dist = zeros((size(distributions)[2], n_transfer_dists * num_pseudotrial_per_dist))
            standard_test_dist = zeros((size(distributions)[2], n_test_dists * num_pseudotrial_per_dist))

            # standard_dist = [zeros((size(distributions)[2], n_dists * num_pseudotrial_per_dist)) for i in 1:standard_nsplits]
            # transfer_dist = [zeros((size(distributions)[2], n_transfer_dists * num_pseudotrial_per_dist)) for i in 1:n_transfers]
            test_dist = zeros((n_transfers, size(distributions)[2], n_test_dists * num_pseudotrial_per_dist))
            # test_dist = zeros((n_transfers, size(distributions)[2], 2 * num_pseudotrial_per_dist))
            # test_dist = zeros((n_transfers, 2 * num_pseudotrial_per_dist))
            test_trial = zeros(Int16, (n_transfers, n_test_dists))

            for session_id_idx in 1:length(session_ids)
                session_id = session_ids[session_id_idx]

                data_session = data_for_selection_session(resp_data, selection_indices, session_id_data, session_id)
                if size(data_session)[2] == 0
                    continue
                end
                trial_numbers = trial_numbers_for_types_in_session(data_session)

                first_idx_in_selected = sum(session_neuron_counts[1:(session_id_idx-1)])+1
                last_idx_in_selected = sum(session_neuron_counts[1:session_id_idx])

                rng = MersenneTwister(d*100000 + session_id_idx)

                # construct pseudotrials for standard dataset, keeping together trials from neurons recorded in same session
                randperm_indices_all = [[] for i in 1:n_dists]
                for trial_idx in 1:n_dists
                    trial_num = trial_numbers[dist_rng[trial_idx]]
                    randperm_indices = randperm(rng, trial_num)
                    randperm_indices_all[trial_idx] = randperm_indices
                    set_selected_num = div(trial_num, standard_nsplits)
                    for pseudo_set_idx in 1:standard_nsplits
                        # randperm_set_indices = randperm_indices[((pseudo_set_idx-1)*set_selected_num+1):(pseudo_set_idx*set_selected_num)]
                        # if pseudo_set_idx == standard_nsplits
                        #     randperm_set_indices = vcat(randperm_set_indices, randperm_indices[(pseudo_set_idx*set_selected_num+1):trial_num])
                        # end
                        # # println(randperm_set_indices)
                        # selected_trial_indices = sample(rng, randperm_set_indices, num_pseudotrial_per_dist, replace=true, ordered=false)
                        # println(selected_trial_indices)
                        selected_trial_indices = sample_trial_indices(pseudo_set_idx, trial_num, randperm_indices, set_selected_num, standard_nsplits, 
                            num_pseudotrial_per_dist, rng)                            
                        standard_data[pseudo_set_idx][first_idx_in_selected:last_idx_in_selected, ((trial_idx-1)*num_pseudotrial_per_dist+1):(trial_idx*num_pseudotrial_per_dist)] =
                            data_session[dist_rng[trial_idx], :, selected_trial_indices]
                    end
                end

                # create pseudotrials for transfer datasets
                all_counts = zeros(Int16, n_dists)
                for (i_set, set) in enumerate(vcat(matched_sets, missed_sets))
                    # for (i_trial, trial_idx) in enumerate(set)
                    train_count = 0
                    test_count = 0
                    for trial_idx in 1:n_dists
                        trial_num = trial_numbers[dist_rng[trial_idx]]
                        if trial_idx in set

                            if trial_idx in [1, 2]
                                nsplits = div(n_transfers, 2)
                                set_count = mod(i_set-1, nsplits) + 1
                            else
                                nsplits = 2  # because trial type [3, 4, 5, 6] appears twice as a training set
                                set_count = mod(all_counts[trial_idx], nsplits) + 1
                            end
                            set_selected_num = div(trial_num, nsplits)

                            train_count += 1
                            all_counts[trial_idx] += 1
                            selected_trial_indices = sample_trial_indices(set_count, trial_num, randperm_indices_all[trial_idx], set_selected_num, nsplits, num_pseudotrial_per_dist, rng)
                            transfer_data[i_set][first_idx_in_selected:last_idx_in_selected, ((train_count-1)*num_pseudotrial_per_dist+1):(train_count*num_pseudotrial_per_dist)] =
                                data_session[dist_rng[trial_idx], :, selected_trial_indices]
                        else
                            test_count += 1
                            selected_trial_indices = sample_trial_indices(1, trial_num, randperm_indices_all[trial_idx], trial_num, 1, num_pseudotrial_per_dist, rng)
                            test_data[i_set][first_idx_in_selected:last_idx_in_selected, ((test_count-1)*num_pseudotrial_per_dist+1):(test_count*num_pseudotrial_per_dist)] = 
                                data_session[dist_rng[trial_idx], :, selected_trial_indices]
                            test_dist[i_set, :, ((test_count-1)*num_pseudotrial_per_dist+1):(test_count*num_pseudotrial_per_dist)] = 
                                distributions[trial_idx, :] * transpose(ones((num_pseudotrial_per_dist)))
                            test_trial[i_set, test_count] = trial_idx
                        end
                    end
                end
            end

            for (nd, store_dist, set) in zip([n_dists, n_transfer_dists, n_test_dists], [standard_dist, transfer_dist, standard_test_dist], [dist_rng, standard_subset, standard_test])
                # construct array of dists
                for dist_idx in 1:nd
                    store_dist[:, ((dist_idx-1)*num_pseudotrial_per_dist+1):(dist_idx*num_pseudotrial_per_dist)] =
                        distributions[set[dist_idx],:] * transpose(ones((num_pseudotrial_per_dist)))
                end
            end
            
            # for dist_idx in 1:n_transfer_dists
            #     transfer_dist[:, ((dist_idx-1)*num_pseudotrial_per_dist+1):(dist_idx*num_pseudotrial_per_dist)] =
            #         distributions[standard_subset[dist_idx],:] * transpose(ones((num_pseudotrial_per_dist)))
            # end

            # for dist_idx in 1:n_test_dists
            #     test_dist[:, ((dist_idx-1)*num_pseudotrial_per_dist+1):(dist_idx*num_pseudotrial_per_dist)] =
            #         distributions[standard_test[dist_idx],:] * transpose(ones((num_pseudotrial_per_dist)))
            # end

            # println("A")
            # println(pseudo_dist[1][:, 1:num_pseudotrial_per_dist:num_pseudotrial_per_dist*n_dists])
            # println(pseudo_dist[pseudo_set_num][:, 1:num_pseudotrial_per_dist:num_pseudotrial_per_dist*n_dists])

            pseudo_file = joinpath(folder, "$(prefix)pseudo_data_$(task_names[d])_cutoff_$(cutoff).jld2")
            @save pseudo_file standard_data transfer_data test_data standard_dist transfer_dist test_dist test_trial
        end

        #####################
        # Training the models
        #####################

        if outer_train

            pseudo_file = joinpath(folder, "$(prefix)pseudo_data_$(task_names[d])_cutoff_$(cutoff).jld2")
            @load pseudo_file standard_data transfer_data test_data standard_dist transfer_dist test_dist test_trial

            # total_set_num = length(pseudo_data)
            # test_set_num = total_set_num - (pseudo_set_num - 1)

            for condition in ["standard", "transfer"]

                println(condition)
                    
                if condition == "standard"
                    n_train_sets = standard_nsplits - 1
                    train_activities = standard_data
                    train_distro = standard_dist
                    # train_dists = standard_dist
                    test_activities = [standard_data[standard_nsplits] for i in 1:n_train_sets]
                    # test_distro = standard_dist
                    test_dists = [standard_dist for i in 1:n_train_sets]
                    
                elseif condition == "transfer"
                    n_train_sets = n_transfers
                    train_activities = transfer_data
                    train_distro = transfer_dist
                    # train_dists = transfer_dist
                    test_activities = test_data
                    # test_dists = test_dist
                    # test_distro = test_dist
                    test_dists = [test_dist[i, :, :] for i in 1:n_transfers]
                    
                end
                
                for set_idx in 1:n_train_sets

                    train_activity = train_activities[set_idx]
                    # train_distro = train_dists[set_idx]
                    test_activity = test_activities[set_idx]
                    test_distro = test_dists[set_idx]

                    # Network
                    res_network_file = joinpath(folder,
                        "$(prefix)res_$(task_names[d])_$(condition)_cutoff_$(cutoff)_set$(set_idx)_network$(num_networkstat).jld2")
                    if network_train
                        println("\nStart training network")
                        train_target_CDFs = dist_to_CDFs(train_distro)
                        n_neurons = size(train_activity)[1]
                        γ = 2e-3
                        if n_neurons > 1700
                            λ = .1
                        else
                            λ = .02
                        end
                        println(γ, " ", λ)
                        W, NW1, NW2, NW3 = minimize_Wasserstein_network(train_activity, reward_values, train_target_CDFs, num_networkstat, 10000, γ, λ)
                        train_Wasserstein = network_Wasserstein_distance_CDFs(W, NW1, NW2, NW3, train_activity, reward_values, train_target_CDFs)
                    else
                        @load res_network_file W NW1 NW2 NW3 train_Wasserstein test_Wasserstein
                    end

                    test_target_CDFs = dist_to_CDFs(test_distro)
                    test_Wasserstein = network_Wasserstein_distance_CDFs(W, NW1, NW2, NW3, test_activity, reward_values, test_target_CDFs)
                    # println("Network, avg Wasserstein distance:  ", train_Wasserstein, "  ", test_Wasserstein)
                    @save res_network_file W NW1 NW2 NW3 train_Wasserstein test_Wasserstein
                end
            end
        end

        #####################
        # Collect the results
        #####################

        if do_collect

            global res

            # results for each train/test pair. standard_nsplits - 1 are from standard; n_transfers are from transfer
            presumed_store_num = standard_nsplits - 1 + n_transfers
            println(presumed_store_num)
            res = fill(NaN, (presumed_store_num, 2, 2))
            standard_decoded_dists = fill(NaN, (standard_nsplits - 1, 2, n_reward_values, n_dists * num_pseudotrial_per_dist))
            transfer_decoded_dists = fill(NaN, (n_transfers, n_reward_values, n_test_dists * num_pseudotrial_per_dist))

            pseudo_file = joinpath(folder, "$(prefix)pseudo_data_$(task_names[d])_cutoff_$(cutoff).jld2")
            @load pseudo_file standard_data transfer_data test_data standard_dist transfer_dist test_dist test_trial

            for condition in ["standard", "transfer"]
                
                if condition == "standard"
                    n_train_sets = standard_nsplits - 1
                    train_activities = standard_data
                    train_distro = standard_dist
                    # train_dists = standard_dist
                    test_activities = [standard_data[standard_nsplits] for i in 1:n_train_sets]
                    # test_distro = standard_dist
                    test_dists = [standard_dist for i in 1:n_train_sets]
                    test_target_CDFs = dist_to_CDFs(standard_dist)
                    
                elseif condition == "transfer"
                    n_train_sets = n_transfers
                    train_activities = transfer_data
                    train_distro = transfer_dist
                    # train_dists = transfer_dist
                    test_activities = test_data
                    # test_distro = test_dist
                    test_dists = [test_dist[i, :, :] for i in 1:n_transfers]
                    test_target_CDFs = dist_to_CDFs(standard_test_dist)
                    
                end

                for set_idx in 1:n_train_sets

                    train_activity = train_activities[set_idx]
                    # train_distro = train_dists[set_idx]
                    test_activity = test_activities[set_idx]
                    test_distro = test_dists[set_idx]

                    if set_idx == 6
                        tmp = 1
                    end

                    # reference performance
                    train_target_CDFs = dist_to_CDFs(train_distro)
                    train_Wasserstein_minimizing_CDF = single_CDF_minimizing_Wasserstein_given_gt_CDFs(transpose(train_target_CDFs))
                    test_Wasserstein_minimizing_CDF = single_CDF_minimizing_Wasserstein_given_gt_CDFs(transpose(test_target_CDFs))
                    train_reference_performance = Wasserstein_distance_given_CDF(train_Wasserstein_minimizing_CDF[1,1:(end-1)], reward_values, train_target_CDFs)
                    test_reference_performance = Wasserstein_distance_given_CDF(test_Wasserstein_minimizing_CDF[1,1:(end-1)], reward_values, test_target_CDFs)
                    println(train_reference_performance," ",test_reference_performance)


                    # Network
                    res_network_file = joinpath(folder,
                        "$(prefix)res_$(task_names[d])_$(condition)_cutoff_$(cutoff)_set$(set_idx)_network$(num_networkstat).jld2")
                    @load res_network_file W NW1 NW2 NW3 train_Wasserstein test_Wasserstein
                    
                    if condition == "standard"
                        res[set_idx, 1, 1] = train_Wasserstein
                        res[set_idx, 1, 2] = train_Wasserstein / train_reference_performance
                        res[set_idx, 2, 1] = test_Wasserstein
                        res[set_idx, 2, 2] = test_Wasserstein / test_reference_performance

                        standard_decoded_dists[set_idx, 1, :, :] = network_decode_dists(W, NW1, NW2, NW3, train_activity)
                        standard_decoded_dists[set_idx, 2, :, :] = network_decode_dists(W, NW1, NW2, NW3, test_activity)

                    elseif condition == "transfer"
                        res[set_idx+standard_nsplits-1, 1, 1] = train_Wasserstein
                        res[set_idx+standard_nsplits-1, 1, 2] = train_Wasserstein / train_reference_performance
                        res[set_idx+standard_nsplits-1, 2, 1] = test_Wasserstein
                        res[set_idx+standard_nsplits-1, 2, 2] = test_Wasserstein / test_reference_performance
                        
                        transfer_decoded_dists[set_idx, :, :] = network_decode_dists(W, NW1, NW2, NW3, test_activity)
                    end
                end
            end

            to_save = [res, standard_decoded_dists, transfer_decoded_dists, test_dist, test_trial]
            save_names = ["result_statistics", "standard_distributions", "transfer_distributions", "true_transfer_distributions", "transfer_trials"]
            for (i_save, (dat, fname)) in enumerate(zip(to_save, save_names))
                # for extension in [".jld2", ".npy"]
                save_path = joinpath(folder, "$(prefix)$(fname)_$(task_names[d])_cutoff_$(cutoff)")
                println("Saving results to path ", save_path)
                @save "$(save_path).jld2" dat
                npzwrite("$(save_path).npy", dat)
            end
        end
    end
end


