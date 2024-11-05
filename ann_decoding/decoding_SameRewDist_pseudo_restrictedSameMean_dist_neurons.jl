include("decoding_utils.jl")


#################
# Hyperparameters
#################

restricted = true

if restricted
    fname = "decoding_SameRewDist_pseudo_restrictedSameMean"
    n_restricted_dists = 4
    restricted_rng = 3:6
else
    fname = "decoding_SameRewDist_pseudo"
    n_restricted_dists = 6
    restricted_rng = collect(1:n_restricted_dists)
end

println("restricted = ", restricted, " fname = ", fname)

# also have the full range, for some things
n_dists = 6
dist_rng = collect(1:n_dists)

#time_window_idx = 2
time_window_idx = 4

# folder of the project
folder = joinpath(folder_path, fname)
if time_window_idx == 2
    prefix = "$(fname)_0to1_"
elseif time_window_idx == 4
    prefix = "$(fname)_2to3_"
end

#datafile_Bernoulli = joinpath(folder_path, "Bernoulli_data.p")
#datafile_odours = joinpath(folder_path, "DistributionalRL_6Odours_data.p")
# datafile_same = joinpath(folder_path, "SameRewDist_6-OHDA_firing_data_20230126.p")
datafile_same = joinpath(folder_path, "SameRewDist_combined_spks_data_20230918.p")


#Bernoulli_data = myunpickle(datafile_Bernoulli)
#odours_data =  myunpickle(datafile_odours)
same_data = myunpickle(datafile_same)


num_networkstat = 16

cutoff = 73
cut_key = "cutoff_" * string(cutoff)
dist_neuron_labels = same_data["neuron_info"][cut_key]
dist_indices = [u for u in 1:length(dist_neuron_labels) if dist_neuron_labels[u]]

pseudo_set_num = 6

num_pseudotrial_per_dist = 100


task_names = ["SameRewDist"]



########################
# Generate pseudo trials
########################


for d in [1]

    data, resp_data, session_id_data, mouse_ids, session_ids, distributions, reward_values, CDFs, CDFs_without_one, constant_intervals, 
        quantiles_on_constant_intervals, regions, dorso_ventral_coordinates = setup_decoding(same_data, d, time_window_idx, dist_rng)

    mouse_session_dict = get_mouse_session_dict(data)
    
    class_names = data["neuron_info"]["helper"]

    for mouse in mouse_ids
        println("\nmouse = ", mouse)
        mouse_sessions = mouse_session_dict[mouse]

        mouse_session_neuron_counts = zeros(Int64,(length(mouse_sessions)))
        mouse_session_selected_neuron_counts = zeros(Int64,(length(mouse_sessions)))

        for session_id_idx in 1:length(mouse_sessions)
            session_id = mouse_sessions[session_id_idx]
            idx = [x for x in 1:size(resp_data)[2] if ((class_names[x] == "all") && (x in dist_indices) &&  (session_id_data[x] == session_id))]
            data_session = resp_data[:,idx,:]
            mouse_session_neuron_counts[session_id_idx] = size(data_session)[2]
        end
        println(mouse_session_neuron_counts, "  ", sum(mouse_session_neuron_counts))

        selected_neuron_indices = 1:sum(mouse_session_neuron_counts)

        pseudo_data = [zeros((length(selected_neuron_indices), size(distributions)[1] * num_pseudotrial_per_dist)) for i in 1:pseudo_set_num]
        pseudo_dist = [zeros((size(distributions)[2], size(distributions)[1] * num_pseudotrial_per_dist)) for i in 1:pseudo_set_num]
        for session_id_idx in 1:length(mouse_sessions)
            session_id = mouse_sessions[session_id_idx]
            idx = [x for x in 1:size(resp_data)[2] if ((class_names[x] == "all") && (x in dist_indices) &&  (session_id_data[x] == session_id))]
            data_session = resp_data[:,idx,:]  # size (n_dists, n_neurons_from_this_session, max_n_trials_per_type)
            if size(data_session)[2] == 0
                continue
            end
            trial_numbers = trial_numbers_for_types_in_session(data_session)

            first_possible_idx = sum(mouse_session_neuron_counts[1:(session_id_idx-1)])+1
            last_possible_idx = sum(mouse_session_neuron_counts[1:session_id_idx])
            selected_session_neuron_indices = [x for x in selected_neuron_indices if (first_possible_idx <= x) && (x <= last_possible_idx)]
            selected_session_neuron_indices = [x - first_possible_idx + 1 for x in selected_session_neuron_indices]

            mouse_session_selected_neuron_counts[session_id_idx] = length(selected_session_neuron_indices)
            first_idx_in_selected = sum(mouse_session_selected_neuron_counts[1:(session_id_idx-1)])+1
            last_idx_in_selected = sum(mouse_session_selected_neuron_counts[1:session_id_idx])

            rng = MersenneTwister(d*100000 + mouse*1000 + session_id_idx)
            for trial_idx in 1:length(trial_numbers)
                trial_num = trial_numbers[trial_idx]
                randperm_indices = randperm(rng, trial_num)
                set_selected_num = div(trial_num, pseudo_set_num)
                for pseudo_set_idx in 1:pseudo_set_num
                    randperm_set_indices = randperm_indices[((pseudo_set_idx-1)*set_selected_num+1):(pseudo_set_idx*set_selected_num)]
                    if pseudo_set_idx == pseudo_set_num
                        randperm_set_indices = vcat(randperm_set_indices, randperm_indices[(pseudo_set_idx*set_selected_num+1):trial_num])
                    end
                    #println(randperm_set_indices)
                    selected_trial_indices = sample(rng, randperm_set_indices, num_pseudotrial_per_dist, replace=true, ordered=false)
                    #println(selected_trial_indices)
                    # pseudo_data gets trials of all types, even when mean_restricted
                    pseudo_data[pseudo_set_idx][first_idx_in_selected:last_idx_in_selected, ((trial_idx-1)*num_pseudotrial_per_dist+1):(trial_idx*num_pseudotrial_per_dist)] =
                        data_session[trial_idx, selected_session_neuron_indices, selected_trial_indices]
                end
            end
        end
        println(mouse_session_selected_neuron_counts, "  ", sum(mouse_session_selected_neuron_counts))

        for pseudo_set_idx in 1:pseudo_set_num
            for dist_idx in 1:size(distributions)[1]
                pseudo_dist[pseudo_set_idx][:, ((dist_idx-1)*num_pseudotrial_per_dist+1):(dist_idx*num_pseudotrial_per_dist)] =
                    distributions[dist_idx,:] * transpose(ones((num_pseudotrial_per_dist)))
            end
        end
        pseudo_file = joinpath(folder, "$(prefix)pseudo_data_$(task_names[d])_mouse$(lpad(mouse,2,"0"))_dist_neurons.jld2")
        @save pseudo_file pseudo_data pseudo_dist
    end
end






#####################
# Training the models
#####################


for d in [1]
    data, resp_data, session_id_data, mouse_ids, session_ids, distributions, reward_values, CDFs, CDFs_without_one, constant_intervals, 
        quantiles_on_constant_intervals, regions, dorso_ventral_coordinates = setup_decoding(same_data, d, time_window_idx, dist_rng)

    class_names = data["neuron_info"]["helper"]

    mouse_session_dict = get_mouse_session_dict(data)

    # indices of the pseudotrials corresponding to the n distributions of the same mean
    restricted_pseudotrial_indices = []
    for dist_idx in restricted_rng
        restricted_pseudotrial_indices = vcat(restricted_pseudotrial_indices, ((dist_idx-1)*num_pseudotrial_per_dist+1):(dist_idx*num_pseudotrial_per_dist))
    end


    for mouse in mouse_ids
        println("\nmouse = ", mouse)
        mouse_sessions = mouse_session_dict[mouse]

        
        println("\nmouse = ", mouse)

        pseudo_file = joinpath(folder, "$(prefix)pseudo_data_$(task_names[d])_mouse$(lpad(mouse,2,"0"))_dist_neurons.jld2")
        if !isfile(pseudo_file)
            continue
        end
        @load pseudo_file pseudo_data pseudo_dist

        for set_idx in 1:(pseudo_set_num-1)
            println("\nmouse = ", mouse, ",   set_idx = ", set_idx)
            train_activity = pseudo_data[set_idx][:, restricted_pseudotrial_indices]
            test_activity = pseudo_data[pseudo_set_num][:, restricted_pseudotrial_indices]
            for i in 1:2
                if i == 1
                    shuffle_indicator = "matched"
                    shuffled_indices = [x for x in 1:n_restricted_dists]
                elseif i == 2
                    shuffle_indicator = "shuffled"
                    rng = MersenneTwister(d*100000 + mouse*1000 + set_idx)
                    shuffled_indices = [x for x in 1:n_restricted_dists]
                    while shuffled_indices == [x for x in 1:n_restricted_dists]
                        shuffled_indices = shuffle(rng, shuffled_indices)
                    end
                end
                println(shuffled_indices)
                train_dist = shuffle_group_of_columns(pseudo_dist[set_idx][:, restricted_pseudotrial_indices], shuffled_indices, num_pseudotrial_per_dist)
                test_dist = shuffle_group_of_columns(pseudo_dist[pseudo_set_num][:, restricted_pseudotrial_indices], shuffled_indices, num_pseudotrial_per_dist)


                println(shuffle_indicator)

                # Quantile code
                println("\nStart training quantile code")
                train_target_quantiles = target_quantiles_given_dist(train_dist, quantiles_on_constant_intervals, distributions)
                test_target_quantiles = target_quantiles_given_dist(test_dist, quantiles_on_constant_intervals, distributions)
                W = minimize_Wasserstein_quantile_code(train_activity, train_target_quantiles, constant_intervals, 10000, 1e-4, 10.0)
                train_Wasserstein = Wasserstein_distance_inverse_CDFs(constant_intervals, W * train_activity, train_target_quantiles)
                test_Wasserstein = Wasserstein_distance_inverse_CDFs(constant_intervals, W * test_activity, test_target_quantiles)
                println("Quantile code, avg Wasserstain distance:  ", train_Wasserstein, "  ", test_Wasserstein)
                res_quantile_file = joinpath(folder, "$(prefix)res_$(task_names[d])_$(shuffle_indicator)_mouse$(lpad(mouse,2,"0"))_dist_neurons_set$(set_idx)_quantile.jld2")
                @save res_quantile_file W train_Wasserstein test_Wasserstein



                # Expectile code
                println("\nStart training expectile code")
                train_target_CDFs = dist_to_CDFs(train_dist)
                test_target_CDFs = dist_to_CDFs(test_dist)
                if d == 1
                    δ = 0.01
                else
                    δ = 0.01
                end
                αs, W = minimize_Wasserstein_expectile_code(train_activity, reward_values, train_target_CDFs, CDFs_without_one[restricted_rng, :], δ, 10000, 1e-7, 10.0)
                train_predicted_CDFs = expectile_decoding_predicted_CDFs(αs, W, train_activity, reward_values, δ)
                train_Wasserstein = Wasserstein_distance_CDFs(reward_values, train_predicted_CDFs, train_target_CDFs)
                test_predicted_CDFs = expectile_decoding_predicted_CDFs(αs, W, test_activity, reward_values, δ)
                test_Wasserstein = Wasserstein_distance_CDFs(reward_values, test_predicted_CDFs, test_target_CDFs)
                println("Expectile code, avg Wasserstain distance:  ", train_Wasserstein, "  ", test_Wasserstein)
                res_expectile_file = joinpath(folder, "$(prefix)res_$(task_names[d])_$(shuffle_indicator)_mouse$(lpad(mouse,2,"0"))_dist_neurons_set$(set_idx)_expectile.jld2")
                @save res_expectile_file αs W train_Wasserstein test_Wasserstein



                # PPC
                println("\nStart training PPC")
                train_target_CDFs = dist_to_CDFs(train_dist)
                test_target_CDFs = dist_to_CDFs(test_dist)
                W = minimize_Wasserstein_PPC(train_activity, reward_values, train_target_CDFs, 10000, 1e-6, 10.0)
                train_Wasserstein = PPC_Wasserstein_distance_CDFs(W, train_activity, reward_values, train_target_CDFs)
                test_Wasserstein = PPC_Wasserstein_distance_CDFs(W, test_activity, reward_values, test_target_CDFs)
                println("PPC, avg Wasserstain distance:  ", train_Wasserstein, "  ", test_Wasserstein)
                res_PPC_file = joinpath(folder, "$(prefix)res_$(task_names[d])_$(shuffle_indicator)_mouse$(lpad(mouse,2,"0"))_dist_neurons_set$(set_idx)_PPC.jld2")
                @save res_PPC_file W train_Wasserstein test_Wasserstein



                # DDC
                println("\nStart training DDC")
                train_target_CDFs = dist_to_CDFs(train_dist)
                test_target_CDFs = dist_to_CDFs(test_dist)
                W = minimize_Wasserstein_DDC(train_activity, reward_values, train_target_CDFs, 10000, 1e-7, 10.0)
                train_Wasserstein = DDC_Wasserstein_distance_CDFs(W, train_activity, reward_values, train_target_CDFs)
                test_Wasserstein = DDC_Wasserstein_distance_CDFs(W, test_activity, reward_values, test_target_CDFs)
                println("DDC, avg Wasserstain distance:  ", train_Wasserstein, "  ", test_Wasserstein)
                res_DDC_file = joinpath(folder, "$(prefix)res_$(task_names[d])_$(shuffle_indicator)_mouse$(lpad(mouse,2,"0"))_dist_neurons_set$(set_idx)_DDC.jld2")
                @save res_DDC_file W train_Wasserstein test_Wasserstein




                # Network
                println("\nStart training network")
                train_target_CDFs = dist_to_CDFs(train_dist)
                test_target_CDFs = dist_to_CDFs(test_dist)
                W, NW1, NW2, NW3 = minimize_Wasserstein_network(train_activity, reward_values, train_target_CDFs, num_networkstat, 10000, 2e-3, 0.02)
                train_Wasserstein = network_Wasserstein_distance_CDFs(W, NW1, NW2, NW3, train_activity, reward_values, train_target_CDFs)
                test_Wasserstein = network_Wasserstein_distance_CDFs(W, NW1, NW2, NW3, test_activity, reward_values, test_target_CDFs)
                println("Network, avg Wasserstain distance:  ", train_Wasserstein, "  ", test_Wasserstein)
                res_network_file = joinpath(folder,
                    "$(prefix)res_$(task_names[d])_$(shuffle_indicator)_mouse$(lpad(mouse,2,"0"))_dist_neurons_set$(set_idx)_network$(num_networkstat).jld2")
                @save res_network_file W NW1 NW2 NW3 train_Wasserstein test_Wasserstein

            end
        end
    end
end









#####################
# Collect the results
#####################


for d in [1]
    data, resp_data, session_id_data, mouse_ids, session_ids, distributions, reward_values, CDFs, CDFs_without_one, constant_intervals, 
        quantiles_on_constant_intervals, regions, dorso_ventral_coordinates = setup_decoding(same_data, d, time_window_idx, dist_rng)
    
    class_names = data["neuron_info"]["helper"]

    mouse_session_dict = get_mouse_session_dict(data)

    # indices of the pseudotrials corresponding to the 3 distributions of the same mean
    restricted_pseudotrial_indices = []
    for dist_idx in restricted_rng
        restricted_pseudotrial_indices = vcat(restricted_pseudotrial_indices, ((dist_idx-1)*num_pseudotrial_per_dist+1):(dist_idx*num_pseudotrial_per_dist))
    end


    for mouse in mouse_ids
        println("\nmouse = ", mouse)
        mouse_sessions = mouse_session_dict[mouse]
        mouse_res = zeros((1, 2, 5, pseudo_set_num-1, 4))

        pseudo_file = joinpath(folder, "$(prefix)pseudo_data_$(task_names[d])_mouse$(lpad(mouse,2,"0"))_dist_neurons.jld2")
        @load pseudo_file pseudo_data pseudo_dist


        for set_idx in 1:(pseudo_set_num-1)
            train_activity = pseudo_data[set_idx][:, restricted_pseudotrial_indices]
            test_activity = pseudo_data[pseudo_set_num][:, restricted_pseudotrial_indices]
            for i in 1:2
                if i == 1
                    shuffle_indicator = "matched"
                    shuffled_indices = [x for x in 1:n_restricted_dists]
                elseif i == 2
                    shuffle_indicator = "shuffled"
                    rng = MersenneTwister(d*100000 + mouse*1000 + set_idx)
                    shuffled_indices = [x for x in 1:n_restricted_dists]
                    while shuffled_indices == [x for x in 1:n_restricted_dists]
                        shuffled_indices = shuffle(rng, shuffled_indices)
                    end
                end
                println(shuffled_indices)
                train_dist = shuffle_group_of_columns(pseudo_dist[set_idx][:, restricted_pseudotrial_indices], shuffled_indices, num_pseudotrial_per_dist)
                test_dist = shuffle_group_of_columns(pseudo_dist[pseudo_set_num][:, restricted_pseudotrial_indices], shuffled_indices, num_pseudotrial_per_dist)


                println(shuffle_indicator)


                # reference performance
                train_target_CDFs = dist_to_CDFs(train_dist)
                test_target_CDFs = dist_to_CDFs(test_dist)
                train_Wasserstein_minimizing_CDF = single_CDF_minimizing_Wasserstein_given_gt_CDFs(transpose(train_target_CDFs))
                test_Wasserstein_minimizing_CDF = single_CDF_minimizing_Wasserstein_given_gt_CDFs(transpose(test_target_CDFs))
                train_reference_performance = Wasserstein_distance_given_CDF(train_Wasserstein_minimizing_CDF[1,1:(end-1)], reward_values, train_target_CDFs)
                test_reference_performance = Wasserstein_distance_given_CDF(test_Wasserstein_minimizing_CDF[1,1:(end-1)], reward_values, test_target_CDFs)
                println(train_reference_performance," ",test_reference_performance)


                # Quantile code
                res_quantile_file = joinpath(folder, "$(prefix)res_$(task_names[d])_$(shuffle_indicator)_mouse$(lpad(mouse,2,"0"))_dist_neurons_set$(set_idx)_quantile.jld2")
                @load res_quantile_file W train_Wasserstein test_Wasserstein
                mouse_res[1, i, 1, set_idx, 1] = train_Wasserstein
                mouse_res[1, i, 1, set_idx, 2] = test_Wasserstein
                mouse_res[1, i, 1, set_idx, 3] = train_Wasserstein / train_reference_performance
                mouse_res[1, i, 1, set_idx, 4] = test_Wasserstein / test_reference_performance


                # Expectile code
                res_expectile_file = joinpath(folder, "$(prefix)res_$(task_names[d])_$(shuffle_indicator)_mouse$(lpad(mouse,2,"0"))_dist_neurons_set$(set_idx)_expectile.jld2")
                @load res_expectile_file αs W train_Wasserstein test_Wasserstein
                mouse_res[1, i, 2, set_idx, 1] = train_Wasserstein
                mouse_res[1, i, 2, set_idx, 2] = test_Wasserstein
                mouse_res[1, i, 2, set_idx, 3] = train_Wasserstein / train_reference_performance
                mouse_res[1, i, 2, set_idx, 4] = test_Wasserstein / test_reference_performance


                # PPC
                res_PPC_file = joinpath(folder, "$(prefix)res_$(task_names[d])_$(shuffle_indicator)_mouse$(lpad(mouse,2,"0"))_dist_neurons_set$(set_idx)_PPC.jld2")
                @load res_PPC_file W train_Wasserstein test_Wasserstein
                mouse_res[1, i, 3, set_idx, 1] = train_Wasserstein
                mouse_res[1, i, 3, set_idx, 2] = test_Wasserstein
                mouse_res[1, i, 3, set_idx, 3] = train_Wasserstein / train_reference_performance
                mouse_res[1, i, 3, set_idx, 4] = test_Wasserstein / test_reference_performance


                # DDC
                res_DDC_file = joinpath(folder, "$(prefix)res_$(task_names[d])_$(shuffle_indicator)_mouse$(lpad(mouse,2,"0"))_dist_neurons_set$(set_idx)_DDC.jld2")
                @load res_DDC_file W train_Wasserstein test_Wasserstein
                mouse_res[1, i, 4, set_idx, 1] = train_Wasserstein
                mouse_res[1, i, 4, set_idx, 2] = test_Wasserstein
                mouse_res[1, i, 4, set_idx, 3] = train_Wasserstein / train_reference_performance
                mouse_res[1, i, 4, set_idx, 4] = test_Wasserstein / test_reference_performance



                # Network
                res_network_file = joinpath(folder,
                    "$(prefix)res_$(task_names[d])_$(shuffle_indicator)_mouse$(lpad(mouse,2,"0"))_dist_neurons_set$(set_idx)_network$(num_networkstat).jld2")
                @load res_network_file W NW1 NW2 NW3 train_Wasserstein test_Wasserstein
                mouse_res[1, i, 5, set_idx, 1] = train_Wasserstein
                mouse_res[1, i, 5, set_idx, 2] = test_Wasserstein
                mouse_res[1, i, 5, set_idx, 3] = train_Wasserstein / train_reference_performance
                mouse_res[1, i, 5, set_idx, 4] = test_Wasserstein / test_reference_performance
            end
        end

        mouse_res_file = joinpath(folder, "$(prefix)result_statistics_$(task_names[d])_mouse$(lpad(mouse,2,"0")).jld2")
        @save mouse_res_file mouse_res

        mouse_res_file2 = joinpath(folder, "$(prefix)result_statistics_$(task_names[d])_mouse$(lpad(mouse,2,"0")).npy")
        npzwrite(mouse_res_file2, mouse_res)
    end
end
