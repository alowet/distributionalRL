using Flux
#using Flux.Tracker
using CUDA
CUDA.allowscalar(true)
using GPUArrays
#using CuArrays
#using CuArrays: @cufunc
#using CUDAdrv
#using CUDAnative
using Juno
using JLD2
using Statistics
using Random
using PoissonRandom
using LinearAlgebra
using Zygote
# using PyPlot
using MAT
using StatsBase
using SpecialFunctions
using NPZ
using DelimitedFiles
using PyCall
using IterTools
# using Distributed
# @pyimport pickle
# @pyimport gzip
pickle = pyimport("pickle")
gzip = pyimport("gzip")
# @pyimport matplotlib.patches as patches

#import Flux.Tracker: Params, gradient, data, update!
import Base.Iterators: enumerate
import Base.depwarn
import NNlib
# import Plots
import Pickle

folder_path = "/home/adam/Documents/dist-rl/code/PPC_DDC_Quantile_Expectile_Comparison"

function myunpickle(filename)
    r = nothing
    @pywith pybuiltin("open")(filename,"rb") as f begin
        r = pickle.load(f)
    end
    return r
end


function data_for_session(resp_data, regions, session_id_data, session_id)  
    idx = [x for x in 1:size(resp_data)[2] if (session_id_data[x] == session_id)]   
    if length(regions) > 0  
        return resp_data[:,idx,:], regions[idx] 
    else    
        return resp_data[:,idx,:], regions  
    end 
end


function data_for_mouse_region(resp_data, mouse_id_list, mouse, regions, reg)   
    idx = [x for x in 1:size(resp_data)[2] if ( (regions[x] == reg) && (mouse_id_list[x] == mouse))]    
    if length(regions) > 0  
        return resp_data[:,idx,:]   
    else    
        return resp_data[:,idx,:]   
    end 
end 


function data_for_class_session_region(resp_data, class_names, class_label, session_id_data, session_id, regions, reg)
    idx = [x for x in 1:size(resp_data)[2] if ((class_names[x] == class_label) && (regions[x] == reg) && (session_id_data[x] == session_id))]
    if length(regions) > 0
        return resp_data[:,idx,:]
    else
        return resp_data[:,idx,:]
    end
end



function data_for_class_session(resp_data, class_names, class_label, session_id_data, session_id)
    idx = [x for x in 1:size(resp_data)[2] if ((class_names[x] == class_label) && (session_id_data[x] == session_id))]
    return resp_data[:,idx,:]
end


function data_for_selection_session(resp_data, selection_indices, session_id_data, session_id)
    idx = [x for x in 1:size(resp_data)[2] if (x in selection_indices) && (session_id_data[x] == session_id)]
    return resp_data[:,idx,:]
end


function trial_numbers_for_types_in_session(data_for_session)
    trial_numbers = zeros(Int, size(data_for_session)[1])
    for i in 1:size(data_for_session)[1]
        idx = size(data_for_session)[3]
        while(isnan(data_for_session[i,1,idx]))
            idx -= 1
        end
        trial_numbers[i] = idx
    end
    return trial_numbers
end

function ground_truth_distributions(dists_data)
    reward_values = []
    for i in 1:length(dists_data)
        reward_values = vcat(reward_values, dists_data[i])
    end
    reward_values = sort(unique(reward_values))
    #println(reward_values)
    res = zeros((length(dists_data), length(reward_values)))
    for i in 1:length(dists_data)
        for r in 1:length(reward_values)
            res[i, r] =  sum(dists_data[i] .== reward_values[r])  / length(dists_data[i])
        end
    end
    return res
end

function ground_truth_reward_values(dists_data)
    reward_values = []
    for i in 1:length(dists_data)
        reward_values = vcat(reward_values, dists_data[i])
    end
    reward_values = sort(unique(reward_values))
    return reward_values
end


function sample_train_test_data(data_for_session, trial_numbers, distributions, rng)
    neuron_num = size(data_for_session)[2]
    reward_value_num = size(distributions)[2]

    train_activity = zeros((neuron_num, 0))
    test_activity = zeros((neuron_num, 0))
    train_dist = zeros((reward_value_num, 0))
    test_dist = zeros((reward_value_num, 0))

    for t in 1:length(trial_numbers)
        trial_num = trial_numbers[t]
        train_num = Int(ceil(trial_num/2))
        test_num = trial_num - train_num

        train_indices = sample(rng, 1:trial_num, train_num, replace=false, ordered=true)
        test_indices = [x for x in 1:trial_num if !(x in train_indices)]

        train_activity = hcat(train_activity, data_for_session[t,:, train_indices])
        test_activity = hcat(test_activity, data_for_session[t,:, test_indices])

        train_dist = hcat(train_dist, distributions[t,:] * transpose(ones((train_num))))
        test_dist = hcat(test_dist, distributions[t,:] * transpose(ones((test_num))))
    end
    return train_activity, test_activity, train_dist, test_dist
end


function partition_train_test_data(data_for_session, trial_numbers, distributions, repetition_num, tt, shuffled_indices)
    neuron_num = size(data_for_session)[2]
    reward_value_num = size(distributions)[2]

    train_activity = zeros((neuron_num, 0))
    test_activity = zeros((neuron_num, 0))
    train_dist = zeros((reward_value_num, 0))
    test_dist = zeros((reward_value_num, 0))

    # train_means = zeros((length(trial_numbers), neuron_num))
    train_means = zeros((neuron_num, 0))
    train_maxs = zeros((neuron_num, 0))

    all_test_indices = []
    all_train_indices = []


    for t in 1:length(trial_numbers)
        trial_num = trial_numbers[t]

        test_indices = [x for x in 1:trial_num if ((x-tt) % repetition_num == 0)]
        train_indices = [x for x in 1:trial_num if !(x in test_indices)]

        push!(all_test_indices, test_indices)
        push!(all_train_indices, train_indices) 

        train_activity = hcat(train_activity, data_for_session[t,:, train_indices])
        test_activity = hcat(test_activity, data_for_session[t,:, test_indices])

        train_mean = mean(data_for_session[t, :, train_indices], dims=2)
        # this will end up being applied to the test set, so we want to tile it that many times
        train_means = hcat(train_means, repeat(train_mean, outer=[1, length(test_indices)]))

        train_max = maximum(data_for_session[t, :, train_indices], dims=2)
        train_maxs = hcat(train_maxs, repeat(train_max, outer=[1, length(test_indices)]))

        v = shuffled_indices[t]

        train_dist = hcat(train_dist, distributions[v,:] * transpose(ones((size(train_indices)[1]))))
        test_dist = hcat(test_dist, distributions[v,:] * transpose(ones((size(test_indices)[1]))))
    end
    return train_activity, test_activity, train_dist, test_dist, train_means, train_maxs, all_test_indices, all_train_indices
end


function shuffle_group_of_columns(data, shuffled_indices, num_pseudotrial_per_dist)
    ans = zeros((size(data)[1], size(data)[2]))
    for t in 1:length(shuffled_indices)
        v = shuffled_indices[t]
        ans[:, ((t-1)*num_pseudotrial_per_dist+1):(t*num_pseudotrial_per_dist)] = data[:, ((v-1)*num_pseudotrial_per_dist+1):(v*num_pseudotrial_per_dist)]
    end
    return ans
end


function shuffle_dist(dist, distributions, shuffled_indices)
    distribution_num = size(distributions)[1]
    ans = zeros(size(dist))
    for k in 1:distribution_num
        v = shuffled_indices[k]
        for c in 1:size(dist)[2]
            if dist[:, c] == distributions[k,:]
                ans[:, c] = distributions[v,:]
            end
        end
    end
    return ans
end

function CDFs_given_distributions(distributions)
    res = ones( (size(distributions)[1], size(distributions)[2]) )
    for i in 1:size(distributions)[1]
        for j in 1:(size(distributions)[2] - 1)
            res[i,j] = sum(distributions[i,1:j])
        end
    end
    return res
end

function single_CDF_minimizing_Wasserstein_given_gt_CDFs(CDFs)
    U = size(CDFs)[1]
    K = size(CDFs)[2]+1
    res = ones((1, K))
    for k in 1:(K-1)
        res[1,k] = median(CDFs[:,k])
    end
    return res
end



function CDFs_curves(CDFs, reward_values)
    U = size(CDFs)[1]
    K = size(CDFs)[2]
    CDFs_x = zeros((U, 2*K))
    CDFs_y = zeros((U, 2*K))
    for u in 1:U
        CDFs_x[u, 1] = reward_values[1]
        CDFs_y[u, 1] = 0.0
        CDFs_x[u, 2] = reward_values[1]
        CDFs_y[u, 2] = CDFs[u, 1]
        for k in 2:K
            CDFs_x[u, 2*k-1] = reward_values[k]
            CDFs_y[u, 2*k-1] = CDFs[u, k-1]
            CDFs_x[u, 2*k] = reward_values[k]
            CDFs_y[u, 2*k] = CDFs[u, k]
        end
    end
    return CDFs_x, CDFs_y
end

function CDFs_given_distributions_without_one(distributions)
    res = zeros( (size(distributions)[1], size(distributions)[2]-1) )
    for i in 1:size(distributions)[1]
        for j in 1:(size(distributions)[2] - 1)
            res[i,j] = sum(distributions[i,1:j])
        end
    end
    return res
end

function constant_inverse_CDF_intervals(CDFs)
    res = sort(unique(CDFs))
    if res[1] != 0.0
        pushfirst!(res, 0.0)
    end
    return res
end

function discrete_quantile_function(q, CDF, reward_values)
    for t in 1:length(reward_values)
        if CDF[t] >= q
            return reward_values[t]
        end
    end
    return reward_values[end]
end

function quantiles_on_constant_inverse_CDF_intervals(CDFs, constant_intervals, reward_values)
    res = zeros( (size(CDFs)[1], length(constant_intervals) - 1) )
    for i in 1:size(CDFs)[1]
        for j in 1:(length(constant_intervals) - 1)
            res[i,j] = discrete_quantile_function(constant_intervals[j+1], CDFs[i,:], reward_values)
        end
    end
    return res
end

function target_quantiles_given_dist(dist, quantiles_on_constant_intervals, distributions)
    res = zeros( (size(quantiles_on_constant_intervals)[2], size(dist)[2]) )
    for j in 1:size(dist)[2]
        for u in 1:size(distributions)[1]
            if dist[:, j] == distributions[u, :]
                res[:,j] = quantiles_on_constant_intervals[u, :]
            end
        end
    end
    return res
end


function Wasserstein_distance_inverse_CDFs(constant_intervals, predicted_quantiles, target_quantiles)
    T = size(predicted_quantiles)[2]
    interval_len = constant_intervals[2:end] .- constant_intervals[1:(end-1)]
    diff = abs.(predicted_quantiles .- target_quantiles)
    return mean( transpose(interval_len) * diff )
end


function Wasserstein_distance_inverse_CDFs_gpu(interval_len, W, activity, target_quantiles)
    diff = CUDA.abs.((W * activity) .- target_quantiles)
    return CUDA.sum( CUDA.transpose(interval_len) * diff )
end



function minimize_Wasserstein_quantile_code(activity, target_quantiles, constant_intervals, iteration=1000, γ=0.01, λ=1.0)
    J = size(activity)[1]
    T = size(activity)[2]
    Q = size(target_quantiles)[1]

    activity_copy = activity

    activity = activity |> gpu
    target_quantiles = target_quantiles |> gpu

    interval_len = constant_intervals[2:end] .- constant_intervals[1:(end-1)]
    if length(interval_len) > 1
        interval_len = reshape(interval_len, (length(interval_len),1))
    else
        interval_len = ones((1,1)) * interval_len[1]
    end
    interval_len = interval_len |> gpu


    W = zeros((Q, J))
    #W = randn!(W)
    W = W |> gpu

    avg_number_terms = (1.0/T)

    for i in 1:iteration
        #println("iteration ",i)
        grad = gradient((W)->(avg_number_terms * Wasserstein_distance_inverse_CDFs_gpu(interval_len, W, activity, target_quantiles)
            + λ*weight_penalty_regularization(W)), W)
        #println(W[4, 1:3:10])
        #println(grad[1][4, 1:3:10])

        W .-= (γ .* (grad[1]))
        #if (i % 1 == 0)
        if (i % 20000 == 0) || (i == 1)
                current_res1 = avg_number_terms * Wasserstein_distance_inverse_CDFs_gpu(interval_len, W, activity, target_quantiles)
                current_res2 = λ*weight_penalty_regularization(W)
                current_res = current_res1 + current_res2
                println("\n iteration ", i, "   ", current_res1,"   ",current_res2)
                println(W[1, 1:4])
                println(grad[1][1, 1:4])
                println(minimum(W |> cpu)," ",maximum(W |> cpu))
                if isnan(current_res)
                        break
                end
        end

        if i % 200 == 0
                # explicitly perform gabarge collection
                GC.gc()
        end
    end

    GC.gc()
    # reclaim cached memory to avoid being out of memory CUDA error
    #CuArrays.reclaim()

    return (W |> cpu)

end


function quantiles_inverse_CDF(quantiles, constant_intervals)
    Q = size(quantiles)[1]
    inverse_CDF_x = zeros((2*Q))
    inverse_CDF_y = zeros((2*Q))
    for q in 1:Q
        inverse_CDF_x[2*q-1] = constant_intervals[q]
        inverse_CDF_y[2*q-1] = quantiles[q]

        inverse_CDF_x[2*q] = constant_intervals[q+1]
        inverse_CDF_y[2*q] = quantiles[q]
    end
    return inverse_CDF_x, inverse_CDF_y
end


function quantiles_validity_rate_and_inverse_CDFs(predicted_quantiles, constant_intervals)
    Q = size(predicted_quantiles)[1]
    T = size(predicted_quantiles)[2]

    inverse_CDFs_x = zeros((2*Q,T))
    inverse_CDFs_y = zeros((2*Q,T))

    global count = 0
    for t in 1:T
        if minimum(predicted_quantiles[2:end, t] .- predicted_quantiles[1:(end-1), t]) >= 0
            global count += 1
        end
        inverse_CDF_x, inverse_CDF_y = quantiles_inverse_CDF(predicted_quantiles[:,t], constant_intervals)
        inverse_CDFs_x[:,t] = inverse_CDF_x
        inverse_CDFs_y[:,t] = inverse_CDF_y
    end
    return count/T, inverse_CDFs_x, inverse_CDFs_y
end



function expectile_decoding_g(α, x, β=10.0)
    return ( α + ( (1.0 - 2*α) / (1 + exp(-1 * β * x)) ) ) * x
end

function expectile_decoding_U_matrix(αs, predicted_expectiles, reward_values)
    K = length(reward_values)
    res = zeros((K-1, K-1))
    for k in 1:(K-1)
        for v in 1:(K-1)
            res[k,v] = expectile_decoding_g(αs[k], predicted_expectiles[k] - reward_values[v]) -
                expectile_decoding_g(αs[k], predicted_expectiles[k] - reward_values[end])
        end
    end
    return res
end



function expectile_decoding_V_vector(αs, predicted_expectiles, reward_values)
    K = length(reward_values)
    res = zeros((K-1))
    for k in 1:(K-1)
            res[k] = expectile_decoding_g(αs[k], predicted_expectiles[k] - reward_values[end])
    end
    return res
end

function expectile_decoding_estimated_p(αs, predicted_expectiles, reward_values, δ=1.0)
    K = length(reward_values)
    #println(αs)
    #println(predicted_expectiles)
    #println(expectile_decoding_U_matrix(αs, predicted_expectiles, reward_values))
    #print("det: ", det(expectile_decoding_U_matrix(αs, predicted_expectiles, reward_values)))
    U = expectile_decoding_U_matrix(αs, predicted_expectiles, reward_values)
    V = expectile_decoding_V_vector(αs, predicted_expectiles, reward_values)
    Id = zeros((K-1, K-1))
    for k in 1:(K-1)
        Id[k,k] = 1.0
    end
    δI = δ .* Id
    #println(αs)
    #println(predicted_expectiles)
    #println(det( (transpose(U) * U ) .+ δI ))
    #println((transpose(U) * V))
    #println( (transpose(U) * U ) .+ δI )
    p = (-1) .* ( inv( (transpose(U) * U ) .+ δI ) * (transpose(U) * V) )
    #println(p)
    return p
end


function expectile_decoding_estimated_ps_gpu(αs, W, activity, reward_values_til_Kminus1, reward_values_end,
        helper_1_Kminus1, helper_1_Kminus1T, helper_Ids_T, helper_1_Kminus1T_sqaure, helper_minusβ_Kminus1T_sqaure, helper_1_0_mask, helper_onehot, helper_δ_Id)

    predicted_expectiles = vec(W * activity)

    α_matrix = helper_Ids_T * αs * helper_1_Kminus1T
    e_matrix = predicted_expectiles * helper_1_Kminus1T
    s_matrix = CUDA.transpose(helper_Ids_T * reward_values_til_Kminus1 * helper_1_Kminus1T)
    reward_values_end_Kminus1 =  helper_1_Kminus1 * reward_values_end
    s_end_matrix = CUDA.transpose(helper_Ids_T * reward_values_end_Kminus1 * helper_1_Kminus1T)

    #=
    intermediate1 = (helper_1_Kminus1T_sqaure .- α_matrix)
    intermediate2 = (α_matrix .+ α_matrix .- helper_1_Kminus1T_sqaure)
    intermediate3 = (e_matrix .- s_matrix)
    intermediate4 = (e_matrix .- s_end_matrix)

    g_on_matrix_1 = ( intermediate1 .+ ( intermediate2 ./
        (helper_1_Kminus1T_sqaure .+ CUDA.exp.( helper_β_Kminus1T_sqaure .* intermediate3 ) ) ) ) .* intermediate3
    g_on_matrix_2 = ( intermediate1 .+ ( intermediate2 ./
        (helper_1_Kminus1T_sqaure .+ CUDA.exp.( helper_β_Kminus1T_sqaure .* intermediate4 ) ) ) ) .* intermediate4
    =#

    intermediate2 = (helper_1_Kminus1T_sqaure .- α_matrix .- α_matrix)
    intermediate3 = (e_matrix .- s_matrix)
    intermediate4 = (e_matrix .- s_end_matrix)

    g_on_matrix_1 = ( α_matrix .+ ( intermediate2 ./
        (helper_1_Kminus1T_sqaure .+ CUDA.exp.( helper_minusβ_Kminus1T_sqaure .* intermediate3 ) ) ) ) .* intermediate3
    g_on_matrix_2 = ( α_matrix .+ ( intermediate2 ./
        (helper_1_Kminus1T_sqaure .+ CUDA.exp.( helper_minusβ_Kminus1T_sqaure .* intermediate4 ) ) ) ) .* intermediate4

    transpose_Us = CUDA.transpose( (g_on_matrix_1 .- g_on_matrix_2) .* helper_1_0_mask )
    minus_Us = (g_on_matrix_2 .- g_on_matrix_1) .* helper_1_0_mask

    Vs = (g_on_matrix_2 * helper_onehot)

    transpose_Us_minus_Us_δ = transpose_Us * minus_Us .- helper_δ_Id

    transpose_Us_Vs = transpose_Us * Vs

    #println("det:", det(minus_Us |> cpu))
    #println("detδ:", det(minus_Us_δ |> cpu))
    #println("minus_Us:", minimum(minus_Us |> cpu)," ", maximum(minus_Us |> cpu))
    #println("minus_Us_δ:", minimum(minus_Us_δ |> cpu)," ", maximum(minus_Us_δ |> cpu))
    #println(Vs)

    ps = transpose_Us_minus_Us_δ \ transpose_Us_Vs
    #println(transpose_Us_minus_Us_δ)
    #println(transpose_Us_Vs)
    #println(ps)

    return ps
end



function expectile_decoding_estimated_Ap(αs, predicted_expectiles, reward_values, δ=1.0)
    K = length(reward_values)
    Y = zeros((K-1, K-1))
    for i in 1:(K-1)
        for j in 1:i
            Y[i,j] = 1.0
        end
    end
    return Y * expectile_decoding_estimated_p(αs, predicted_expectiles, reward_values, δ)
end


function expectile_decoding_predicted_CDFs(αs, W, activity, reward_values, δ=1.0)
    K = length(reward_values)
    T = size(activity)[2]

    res = zeros((K-1, T))
    for t in 1:T
        res[:, t] = expectile_decoding_estimated_Ap(αs, W * activity[:, t], reward_values, δ)
    end
    return res
end

function expectile_decoding_predicted_CDFs_gpu(αs, W, activity, reward_values_til_Kminus1, reward_values_end,
        helper_1_Kminus1, helper_1_Kminus1T, helper_Ids_T, helper_1_Kminus1T_sqaure, helper_minusβ_Kminus1T_sqaure, helper_1_0_mask, helper_onehot, helper_δ_Id,
        helper_accumulation)

        K = size(W)[1] + 1
        T = size(activity)[2]

        ps = expectile_decoding_estimated_ps_gpu(αs, W, activity, reward_values_til_Kminus1, reward_values_end,
                helper_1_Kminus1, helper_1_Kminus1T, helper_Ids_T, helper_1_Kminus1T_sqaure, helper_minusβ_Kminus1T_sqaure, helper_1_0_mask, helper_onehot, helper_δ_Id)
        ps = CUDA.reshape(ps, (K-1, T))
        #println("ps:", ps)
        CDFs = helper_accumulation * ps
        #println("CDFs:", CDFs)
        return CDFs
end

function dist_to_CDFs(dist)
    L = size(dist)[1]
    Y = zeros((L-1,L))
    for l in 1:(L-1)
        for m in 1:l
            Y[l, m] = 1.0
        end
    end
    return Y*dist
end



function Wasserstein_distance_CDFs(reward_values, predicted_CDFs, target_CDFs)
    K = length(reward_values)
    reward_values_diff = reward_values[2:end] .- reward_values[1:(end-1)]
    diff = abs.(predicted_CDFs .- target_CDFs)
    return mean( transpose(reward_values_diff) * diff)
end


function expectile_Wasserstein_distance_CDFs_gpu(αs, W, activity, reward_values_til_Kminus1, reward_values_end, reward_values_diff, target_CDFs,
        helper_1_Kminus1, helper_1_Kminus1T, helper_Ids_T, helper_1_Kminus1T_sqaure, helper_minusβ_Kminus1T_sqaure, helper_1_0_mask, helper_onehot, helper_δ_Id,
        helper_accumulation)
    predicted_CDFs = expectile_decoding_predicted_CDFs_gpu(αs, W, activity, reward_values_til_Kminus1, reward_values_end,
            helper_1_Kminus1, helper_1_Kminus1T, helper_Ids_T, helper_1_Kminus1T_sqaure, helper_minusβ_Kminus1T_sqaure, helper_1_0_mask, helper_onehot,
            helper_δ_Id, helper_accumulation)
    diff = CUDA.abs.(predicted_CDFs .- target_CDFs)
    #println("predicted_CDFs: ", predicted_CDFs)
    #println("target_CDFs: ", target_CDFs)
    #println("diff: ", diff)
    #println("reward_values_diff: ", reward_values_diff)
    return CUDA.sum( CUDA.transpose(reward_values_diff) * diff)
end



function minimize_Wasserstein_expectile_code(activity, reward_values, target_CDFs, CDFs_without_one, δ, iteration=1000, γ=0.01, λ=1.0)
        J = size(activity)[1]
        #T = size(activity)[2]
        T_original = size(activity)[2]
        T = 4
        K = length(reward_values)
        U = size(CDFs_without_one)[1]
        println(J," ",T_original," ",K," ", U)

        activity_copy = activity
        target_CDFs_copy = target_CDFs

        activity = activity |> gpu

        reward_values_til_Kminus1 = reward_values[1:(end-1)]
        if length(reward_values_til_Kminus1) > 1
            reward_values_til_Kminus1 = reshape(reward_values_til_Kminus1, (length(reward_values_til_Kminus1),1))
        else
            reward_values_til_Kminus1 = ones((1,1)) * reward_values_til_Kminus1[1]
        end
        reward_values_til_Kminus1 = reward_values_til_Kminus1 |> gpu

        reward_values_end = reward_values[end:end]
        reward_values_end = ones((1,1)) * reward_values_end[1]
        reward_values_end = reward_values_end |> gpu

        reward_values_diff = reward_values[2:end] .- reward_values[1:(end-1)]
        if length(reward_values_diff) > 1
            reward_values_diff = reshape(reward_values_diff, (length(reward_values_diff),1))
        else
            reward_values_diff = ones((1,1)) * reward_values_diff[1]
        end
        reward_values_diff = reward_values_diff |> gpu

        target_CDFs = target_CDFs |> gpu

        helper_1_Kminus1 = ones((K-1,1))
        helper_1_Kminus1 = helper_1_Kminus1 |> gpu
        helper_1_Kminus1T = ones((1,(K-1)*T))
        helper_1_Kminus1T = helper_1_Kminus1T |> gpu

        helper_Ids_T = zeros(((K-1)*T, K-1))
        for t in 1:T
            for k in 1:(K-1)
                helper_Ids_T[(K-1)*(t-1) + k, k] = 1.0
            end
        end
        helper_Ids_T = helper_Ids_T |> gpu

        helper_1_Kminus1T_sqaure = ones(((K-1)*T, (K-1)*T))
        minusβ = -10.0
        helper_minusβ_Kminus1T_sqaure = minusβ .* helper_1_Kminus1T_sqaure
        helper_1_Kminus1T_sqaure = helper_1_Kminus1T_sqaure |> gpu
        helper_minusβ_Kminus1T_sqaure = helper_minusβ_Kminus1T_sqaure |> gpu

        helper_1_0_mask = zeros(((K-1)*T, (K-1)*T))
        for t in 1:T
            helper_1_0_mask[((K-1)*(t-1) + 1):((K-1)*t), ((K-1)*(t-1) + 1):((K-1)*t)] = ones((K-1, K-1))
        end
        helper_1_0_mask = helper_1_0_mask |> gpu

        helper_onehot = zeros(((K-1)*T, 1))
        helper_onehot[end, 1] = 1.0
        helper_onehot = helper_onehot |> gpu

        helper_δ_Id = zeros(((K-1)*T, (K-1)*T))
        for z in 1:((K-1)*T)
            helper_δ_Id[z,z] = δ
        end
        helper_δ_Id = helper_δ_Id |> gpu

        helper_accumulation = zeros((K-1, K-1))
        for k1 in 1:(K-1)
            for k2 in 1:k1
                helper_accumulation[k1, k2] = 1.0
            end
        end
        helper_accumulation = helper_accumulation |> gpu


        avg_number_terms = (1.0/T)

        γα =  1 * γ
        γW = γ

        rng = MersenneTwister(1)

        println("selecting candidates: ")
        candidate_num = 50
        α_candidates = Array{Float64,2}(undef, (candidate_num, K-1))
        W_candidates = Array{Float64,3}(undef, (candidate_num, K-1, J))
        result_candidates = Array{Float64,1}(undef, candidate_num)
        ϵ = 0.0
        for v in 1:candidate_num

                α_v = Array{Float64,1}(undef, K-1)
                rand!(α_v)
                α_v = sort(α_v)
                #=
                if K > 2
                    while minimum(α_v[2:end] .- α_v[1:(end-1)]) < 0.01
                        rand!(α_v)
                        α_v = sort(α_v)
                    end
                end
                =#

                #min_total_activity = minimum(sum(activity_copy, dims=1))
                #max_total_activity = maximum(sum(activity_copy, dims=1))
                #println("total activity: ",min_total_activity," ",max_total_activity)
                #W_v = ones((K-1, J)) .* (reward_values[end] / (max(min_total_activity, 0) + max_total_activity))
                #W_v = Array{Float64,2}(undef, (K-1, J))
                #randn!(W_v)
                #W_v = W_v ./ 100
                #W_v = ones((K-1, J)) .* (reward_values_mean / (activity_mean * J))
                #W_v = zeros((K-1, J))

                #simple_target_expectiles = ones((K-1, T))

                #=
                simple_target_expectiles = zeros((K-1, T))
                for k in 1:(K-1)
                    for t in 1:T
                        for u in 1:U

                            if CDFs_without_one[u,:] == target_CDFs_copy[:,t]
                                #simple_target_expectiles[k,:] .*= (reward_values[1] + (k/K)*(reward_values[end] - reward_values[1]))
                                simple_target_expectiles[k,t] = gt_E[u, Int(ceil(α_v[k]*100))]
                                #simple_target_expectiles[k,t] = gt_E[u, 50]
                            end

                        end
                    end
                end

                #println(simple_target_expectiles)
                W_smart_start = simple_target_expectiles / activity_copy
                W_v = W_smart_start
                println("")
                println(minimum(simple_target_expectiles)," ",maximum(simple_target_expectiles))
                println(minimum(W_v)," ",maximum(W_v))
                println(minimum((W_v * activity_copy) .- simple_target_expectiles)," ",
                    maximum((W_v * activity_copy) .- simple_target_expectiles))
                println(minimum((W_v * activity_copy) )," ",
                    maximum((W_v * activity_copy)))


                if (minimum( W_v * activity_copy ) < (reward_values[1] - ϵ) ) || (maximum( W_v * activity_copy ) > (reward_values[end] + ϵ) )
                    min_total_activity = minimum(sum(activity_copy, dims=1))
                    max_total_activity = maximum(sum(activity_copy, dims=1))
                    println("total activity: ",min_total_activity," ",max_total_activity)
                    W_v = ones((K-1, J)) .* (reward_values[end] / (min_total_activity + max_total_activity))

                    println("backup initialization")
                    println(minimum(W_v)," ",maximum(W_v))
                    println(minimum((W_v * activity_copy) .- simple_target_expectiles)," ",
                        maximum((W_v * activity_copy) .- simple_target_expectiles))
                    println(minimum((W_v * activity_copy) )," ",
                        maximum((W_v * activity_copy)))
                end
                =#

                min_total_activity = minimum(sum(activity_copy, dims=1))
                max_total_activity = maximum(sum(activity_copy, dims=1))
                mean_total_activity = mean(sum(activity_copy, dims=1))
                #println("\ntotal activity: ",min_total_activity," ",mean_total_activity," ",max_total_activity)
                W_v = zeros((K-1, J))
                while true
                    randn!(W_v)
                    W_v .*= (1/1000)
                    #W_v .+= ( 2 * mean(reward_values) / (min_total_activity + max_total_activity))
                    for k in 1:(K-1)
                        W_v[k,:] .+= ( (k/K) * reward_values[end] / max_total_activity )
                    end
                    ((minimum( W_v * activity_copy ) < reward_values[1] ) || (maximum( W_v * activity_copy ) > reward_values[end] )) || break
                end

                #println(minimum(W_v)," ",maximum(W_v))
                #println(minimum((W_v * activity_copy) .- simple_target_expectiles)," ",
                #    maximum((W_v * activity_copy) .- simple_target_expectiles))
                #println(minimum((W_v * activity_copy) )," ",
                #    maximum((W_v * activity_copy)))

                #=
                #println(size(simple_target_expectiles)," ",size(activity_copy)," ",size(ones((K-1,K-1))))
                W_smart_start = simple_target_expectiles * transpose(activity_copy) *
                    inv((activity_copy * transpose(activity_copy)) .+ (0.0001 .* ones((J,J))) )
                W_v = W_smart_start
                println(minimum(W_v)," ",maximum(W_v))
                println(minimum((W_v * activity_copy) .- simple_target_expectiles)," ",
                    maximum((W_v * activity_copy) .- simple_target_expectiles))
                =#
                #println(maximum(W_v)," ",mean(W_v)," ",maximum(W_v))




                α_candidates[v,:] = α_v
                W_candidates[v,:,:] = W_v

                α_v = α_v |> gpu
                W_v = W_v |> gpu


                for w in 1:200
                        sampled_indices = sample(rng, 1:T_original, T, replace=false, ordered=true)
                        grad = gradient((α_v, W_v)->(avg_number_terms * expectile_Wasserstein_distance_CDFs_gpu(
                                α_v, W_v, activity[:,sampled_indices], reward_values_til_Kminus1, reward_values_end, reward_values_diff, target_CDFs[:,sampled_indices],
                                helper_1_Kminus1, helper_1_Kminus1T, helper_Ids_T, helper_1_Kminus1T_sqaure, helper_minusβ_Kminus1T_sqaure,
                                helper_1_0_mask, helper_onehot, helper_δ_Id, helper_accumulation) + λ*weight_penalty_regularization(W_v) ),
                                α_v, W_v)
                        α_v .-= (γα .* (grad[1]))
                        W_v .-= (γW .* (grad[2]))
                        if (w % 1000 == 0)
                        #if (w % 100 == 0) || (w == 1)
                            current_res1 = avg_number_terms * expectile_Wasserstein_distance_CDFs_gpu(
                                    α_v, W_v, activity[:,sampled_indices], reward_values_til_Kminus1, reward_values_end, reward_values_diff, target_CDFs[:,sampled_indices],
                                    helper_1_Kminus1, helper_1_Kminus1T, helper_Ids_T, helper_1_Kminus1T_sqaure, helper_minusβ_Kminus1T_sqaure,
                                    helper_1_0_mask, helper_onehot, helper_δ_Id, helper_accumulation)
                            current_res2 = λ*weight_penalty_regularization(W_v)
                            println("\n iteration ", w, "   ", current_res1,"   ",current_res2)
                            println( minimum((W_v * activity[:,sampled_indices]) |> cpu)," ", maximum((W_v * activity[:,sampled_indices]) |> cpu) )
                            println(α_v)
                            println(grad[1])
                            println(W_v[1,:1:4])
                            println(grad[2][1,1:4])
                            println(minimum(W_v |> cpu)," ",maximum(W_v |> cpu))
                        end
                        if w % 100 == 0
                                GC.gc()
                                #CuArrays.reclaim()
                        end
                end
                #=
                result_candidates[v] = avg_number_terms * expectile_Wasserstein_distance_CDFs_gpu(
                        α_v, W_v, activity, reward_values_til_Kminus1, reward_values_end, reward_values_diff, target_CDFs,
                        helper_1_Kminus1, helper_1_Kminus1T, helper_Ids_T, helper_1_Kminus1T_sqaure, helper_minusβ_Kminus1T_sqaure,
                        helper_1_0_mask, helper_onehot, helper_δ_Id, helper_accumulation) + λ*weight_penalty_regularization(W_v)
                =#

                candidate_predicted_CDFs = expectile_decoding_predicted_CDFs((α_v |> cpu), (W_v |> cpu), activity_copy, reward_values, δ)
                candidate_Wasserstein = Wasserstein_distance_CDFs(reward_values, candidate_predicted_CDFs, target_CDFs_copy)
                result_candidates[v] = candidate_Wasserstein




                if v % 1 == 0
                        print("$(v) ")
                        print(result_candidates[v]," ")
                end

        end

        argmax_idx = argmin_not_NaN(result_candidates)
        println("selected candidate preresult: ",result_candidates[argmax_idx])
        α = α_candidates[argmax_idx,:] |> gpu
        W = W_candidates[argmax_idx,:,:] |> gpu

        GC.gc()
        #CuArrays.reclaim()


        for i in 1:iteration
                #println("iteration ",i)
                sampled_indices = sample(rng, 1:T_original, T, replace=false, ordered=true)
                grad = gradient((α, W)->(avg_number_terms * expectile_Wasserstein_distance_CDFs_gpu(
                        α, W, activity[:,sampled_indices], reward_values_til_Kminus1, reward_values_end, reward_values_diff, target_CDFs[:,sampled_indices],
                        helper_1_Kminus1, helper_1_Kminus1T, helper_Ids_T, helper_1_Kminus1T_sqaure, helper_minusβ_Kminus1T_sqaure,
                        helper_1_0_mask, helper_onehot, helper_δ_Id, helper_accumulation) + λ*weight_penalty_regularization(W) ),
                        α, W)

                α .-= (γα .* (grad[1]))
                W .-= (γW .* (grad[2]))

                if (i % 20000 == 0) || (i == 1)
                        current_res1 = avg_number_terms * expectile_Wasserstein_distance_CDFs_gpu(
                                α, W, activity[:,sampled_indices], reward_values_til_Kminus1, reward_values_end, reward_values_diff, target_CDFs[:,sampled_indices],
                                helper_1_Kminus1, helper_1_Kminus1T, helper_Ids_T, helper_1_Kminus1T_sqaure, helper_minusβ_Kminus1T_sqaure,
                                helper_1_0_mask, helper_onehot, helper_δ_Id, helper_accumulation)
                        current_res2 = λ*weight_penalty_regularization(W)
                        current_res = current_res1 + current_res2
                        println("\n iteration ", i, "   ", current_res1,"   ", current_res2)
                        println( minimum((W * activity[:,sampled_indices]) |> cpu)," ", maximum((W * activity[:,sampled_indices]) |> cpu) )
                        println(α)
                        println(grad[1])
                        println(W[1,:1:4])
                        println(grad[2][1,1:4])
                        println(minimum(W |> cpu)," ",maximum(W |> cpu))
                        if isnan(current_res)
                                break
                        end
                end

                if i % 200 == 0
                        # explicitly perform gabarge collection
                        GC.gc()
                end
        end

        GC.gc()
        # reclaim cached memory to avoid being out of memory CUDA error
        #CuArrays.reclaim()

        return (α |> cpu), (W |> cpu)

end



function argmin_not_NaN(a)
        aa = [x for x in a if !isnan(x)]
        if length(aa) == 0
                return -99999
        else
                L = size(a)[1]
                res_idx = 0
                res_max = aa[1]
                for l in 1:L
                        if (!isnan(a[l])) && (a[l] <= res_max)
                                res_idx = l
                                res_max = a[l]
                        end
                end
                return res_idx
        end
end



function PPC_decode_CDFs2(W, activity)
    K = size(W)[1]
    J = size(W)[2]
    T = size(activity)[2]

    natural_params = W * activity
    exp_natural_params = exp.(natural_params)
    sums = sum( exp_natural_params, dims=1 )

    decoded_probas = zeros((K,T))
    for k in 1:K
        for t in 1:T
            decoded_probas[k, t] = exp_natural_params[k,t] / sums[t]
        end
    end
    Y = zeros((K, K))
    for i in 1:K
        for j in 1:i
            Y[i,j] = 1.0
        end
    end
    predicted_CDFs2 = Y * decoded_probas

    return predicted_CDFs2
end




function PPC_Wasserstein_distance_CDFs(W, activity, reward_values, target_CDFs)
    K = size(W)[1]
    J = size(W)[2]
    T = size(activity)[2]

    natural_params = W * activity
    exp_natural_params = exp.(natural_params)
    sums = sum( exp_natural_params, dims=1 )

    decoded_probas = zeros((K,T))
    for k in 1:K
        for t in 1:T
            decoded_probas[k, t] = exp_natural_params[k,t] / sums[t]
        end
    end
    Y = zeros((K-1, K))
    for i in 1:(K-1)
        for j in 1:i
            Y[i,j] = 1.0
        end
    end
    predicted_CDFs = Y * decoded_probas

    reward_values_diff = reward_values[2:end] .- reward_values[1:(end-1)]
    diff = abs.(predicted_CDFs .- target_CDFs)
    return mean(transpose(reward_values_diff) * diff)
end


function PPC_Wasserstein_distance_CDFs_gpu(W, activity, reward_values_diff, target_CDFs, helper_1_K, helper_accumulation)
    natural_params = W * activity
    exp_natural_params = CUDA.exp.(natural_params)
    sums = CUDA.sum( exp_natural_params, dims=1 )
    sums_matrix = helper_1_K * sums
    decoded_probas = exp_natural_params ./ sums_matrix
    predicted_CDFs = helper_accumulation * decoded_probas

    diff = CUDA.abs.(predicted_CDFs .- target_CDFs)
    return CUDA.sum(CUDA.transpose(reward_values_diff) * diff)
end



function minimize_Wasserstein_PPC(activity, reward_values, target_CDFs, iteration=1000, γ=0.01, λ=1.0)
        J = size(activity)[1]
        T = size(activity)[2]
        K = length(reward_values)

        activity = activity |> gpu

        reward_values_diff = reward_values[2:end] .- reward_values[1:(end-1)]
        if length(reward_values_diff) > 1
            reward_values_diff = reshape(reward_values_diff, (length(reward_values_diff),1))
        else
            reward_values_diff = ones((1,1)) * reward_values_diff[1]
        end
        reward_values_diff = reward_values_diff |> gpu

        target_CDFs = target_CDFs |> gpu

        helper_1_K = ones((K,1))
        helper_1_K = helper_1_K |> gpu

        helper_accumulation = zeros((K-1, K))
        for k1 in 1:(K-1)
            for k2 in 1:k1
                helper_accumulation[k1, k2] = 1.0
            end
        end
        helper_accumulation = helper_accumulation |> gpu


        avg_number_terms = (1.0/T)


        println("selecting candidates: ")
        candidate_num = 50
        W_candidates = Array{Float64,3}(undef, (candidate_num, K, J))
        result_candidates = Array{Float64,1}(undef, candidate_num)

        for v in 1:candidate_num

                W_v = Array{Float64,2}(undef, (K, J))
                randn!(W_v)
                W_v = W_v ./= 100

                W_candidates[v,:,:] = W_v

                W_v = W_v |> gpu


                for w in 1:200
                        grad = gradient((W_v)->( avg_number_terms *
                                PPC_Wasserstein_distance_CDFs_gpu(W_v, activity, reward_values_diff, target_CDFs, helper_1_K, helper_accumulation)
                                + λ*weight_penalty_regularization(W_v) ),
                                W_v)
                        W_v .-= (γ .* (grad[1]))
                        if w % 100 == 0
                                GC.gc()
                                #CuArrays.reclaim()
                        end
                end
                result_candidates[v] = avg_number_terms *
                    PPC_Wasserstein_distance_CDFs_gpu(W_v, activity, reward_values_diff, target_CDFs, helper_1_K, helper_accumulation)
                    + λ*weight_penalty_regularization(W_v)

                if v % 1 == 0
                        print("$(v) ")
                        print(result_candidates[v]," ")
                end

        end

        argmax_idx = argmin_not_NaN(result_candidates)
        println("selected candidate preresult: ",result_candidates[argmax_idx])
        W = W_candidates[argmax_idx,:,:] |> gpu


        GC.gc()
        #CuArrays.reclaim()



        for i in 1:iteration
                #println("iteration ",i)
                grad = gradient((W)->( avg_number_terms *
                        PPC_Wasserstein_distance_CDFs_gpu(W, activity, reward_values_diff, target_CDFs, helper_1_K, helper_accumulation)
                        + λ*weight_penalty_regularization(W) ),
                        W)
                W .-= (γ .* (grad[1]))
                if (i % 20000 == 0) || (i == 1)
                        current_res1 = avg_number_terms *
                                PPC_Wasserstein_distance_CDFs_gpu(W, activity, reward_values_diff, target_CDFs, helper_1_K, helper_accumulation)
                        current_res2 = λ*weight_penalty_regularization(W)
                        current_res = current_res1 + current_res2
                        println("\n iteration ", i, "   ", current_res1,"   ", current_res2)
                        println(W[1,1:4])
                        println(grad[1][1,1:4])
                        println(minimum(W |> cpu)," ",maximum(W |> cpu))
                        if isnan(current_res)
                                break
                        end
                end

                if i % 200 == 0
                        # explicitly perform gabarge collection
                        GC.gc()
                end

        end

        GC.gc()
        # reclaim cached memory to avoid being out of memory CUDA error
        #CuArrays.reclaim()

        return (W |> cpu)

end



function DDC_decode_CDFs2(W, activity)
    K = size(W)[1] + 1
    J = size(W)[2]
    T = size(activity)[2]

    decoded_probas = W * activity
    Y = zeros((K-1, K-1))
    for i in 1:(K-1)
        for j in 1:i
            Y[i,j] = 1.0
        end
    end
    predicted_CDFs = Y * decoded_probas

    predicted_CDFs2 = vcat(predicted_CDFs, transpose([1.0 for i in 1:T]))

    return predicted_CDFs2
end


function DDC_Wasserstein_distance_CDFs(W, activity, reward_values, target_CDFs)
    K = size(W)[1] + 1
    J = size(W)[2]
    T = size(activity)[2]

    decoded_probas = W * activity
    Y = zeros((K-1, K-1))
    for i in 1:(K-1)
        for j in 1:i
            Y[i,j] = 1.0
        end
    end
    predicted_CDFs = Y * decoded_probas

    reward_values_diff = reward_values[2:end] .- reward_values[1:(end-1)]
    diff = abs.(predicted_CDFs .- target_CDFs)
    return mean(transpose(reward_values_diff) * diff)
end



function DDC_Wasserstein_distance_CDFs_gpu(W, activity, reward_values_diff, target_CDFs, helper_accumulation)
    decoded_probas = W * activity
    predicted_CDFs = helper_accumulation * decoded_probas

    diff = CUDA.abs.(predicted_CDFs .- target_CDFs)
    return CUDA.sum(CUDA.transpose(reward_values_diff) * diff)
end



function minimize_Wasserstein_DDC(activity, reward_values, target_CDFs, iteration=1000, γ=0.01, λ=1.0)
    J = size(activity)[1]
    T = size(activity)[2]
    K = length(reward_values)

    activity_copy = activity

    activity = activity |> gpu

    reward_values_diff = reward_values[2:end] .- reward_values[1:(end-1)]
    if length(reward_values_diff) > 1
        reward_values_diff = reshape(reward_values_diff, (length(reward_values_diff),1))
    else
        reward_values_diff = ones((1,1)) * reward_values_diff[1]
    end
    reward_values_diff = reward_values_diff |> gpu

    target_CDFs = target_CDFs |> gpu

    helper_accumulation = zeros((K-1, K-1))
    for k1 in 1:(K-1)
        for k2 in 1:k1
            helper_accumulation[k1, k2] = 1.0
        end
    end
    helper_accumulation = helper_accumulation |> gpu


    min_total_activity = minimum(sum(activity_copy, dims=1))
    max_total_activity = maximum(sum(activity_copy, dims=1))
    mean_total_activity = mean(sum(activity_copy, dims=1))

    W = ones((K-1, J)) .* ( (1/K) / max_total_activity )
    W = W |> gpu

    avg_number_terms = (1.0/T)

    for i in 1:iteration
        #println("iteration ",i)
        grad = gradient((W)->(avg_number_terms * DDC_Wasserstein_distance_CDFs_gpu(W, activity, reward_values_diff, target_CDFs, helper_accumulation)
            + λ*weight_penalty_regularization(W) ), W)

        W .-= (γ .* (grad[1]))
        if (i % 20000 == 0) || (i == 1)
                current_res1 = avg_number_terms * DDC_Wasserstein_distance_CDFs_gpu(W, activity, reward_values_diff, target_CDFs, helper_accumulation)
                current_res2 = λ*weight_penalty_regularization(W)
                current_res = current_res1 + current_res2
                println("\n iteration ", i, "   ", current_res1, "   ", current_res2)
                println(W[1, 1:4])
                println(grad[1][1, 1:4])
                println(minimum(W |> cpu)," ",maximum(W |> cpu))
                if isnan(current_res)
                        break
                end
        end

        if i % 200 == 0
                # explicitly perform gabarge collection
                GC.gc()
        end
    end

    GC.gc()
    # reclaim cached memory to avoid being out of memory CUDA error
    #CuArrays.reclaim()

    return (W |> cpu)

end






function ELU_approximate(x)
    return log(1+exp(x+1)) - 1
end

function ELU_approximate_gpu(x, helper_1)
    return CUDA.log.( helper_1 .+ CUDA.exp.(x .+ helper_1) ) .- helper_1
end

function input_to_networkstat(NW1, NW2, NW3, input)
    layer1_output = ELU_approximate.(NW1 * input)
    layer2_output = ELU_approximate.(NW2 * layer1_output)
    layer3_output = ELU_approximate.(NW3 * layer2_output)
    return layer3_output
end

function input_to_networkstat_gpu(NW1, NW2, NW3, input, helper_1_layer1, helper_1_layer2, helper_1_layer3)
    layer1_output = ELU_approximate_gpu(NW1 * input, helper_1_layer1)
    layer2_output = ELU_approximate_gpu(NW2 * layer1_output, helper_1_layer2)
    layer3_output = ELU_approximate_gpu(NW3 * layer2_output, helper_1_layer3)
    return layer3_output
end


function dists_to_networkstat_to_predicted_activity(NW1, NW2, NW3, M, networkstat_to_trials_multiplier, dists)
    stats = input_to_networkstat(NW1, NW2, NW3, dists)
    predicted_activity = M * stats * networkstat_to_trials_multiplier
    return predicted_activity
end

function dists_to_networkstat_to_predicted_activity_gpu(NW1, NW2, NW3, M, networkstat_to_trials_multiplier, dists,
        helper_1_layer1, helper_1_layer2, helper_1_layer3)
    stats = input_to_networkstat_gpu(NW1, NW2, NW3, dists, helper_1_layer1, helper_1_layer2, helper_1_layer3)
    predicted_activity = M * stats * networkstat_to_trials_multiplier
    return predicted_activity
end



function log_likelihood_networkstat(NW1, NW2, NW3, M, networkstat_to_trials_multiplier, dists, activity, activity_mask)
    J = size(activity)[1]
    L = size(dists)[2]
    T = size(activity)[2]

    predicted_activity = dists_to_networkstat_to_predicted_activity(NW1, NW2, NW3, M, networkstat_to_trials_multiplier, dists)

    avg = 1/sum(activity_mask)
    res = 0.0
    for j in 1:J
            for t in 1:T
                    λ = predicted_activity[j,t]
                    if λ >= 0
                            fλ = (1/100.0)*(100*λ+log(1+exp(-100*λ)))
                    else
                            fλ = (1/100.0)*(log(1+exp(100*λ)))
                    end

                    if activity_mask[j,t] ==  1.0
                            r = activity[j,t]
                            # if λ is too small or even negative, deliberately make the likelihood small
                            #res += avg*(λ > 0.001 ? (r*log(λ) - λ -log(factorial(r))) :
                            #        (r*log(0.001) - 0.001 -log(factorial(r)) + 10000*(λ-0.001)) )
                            res += avg*(r*log(fλ) - fλ)
                    end
            end
    end
    return res

end


function log_likelihood_networkstat_gpu(NW1, NW2, NW3, M, networkstat_to_trials_multiplier, dists, activity, activity_mask,
        helper_1_layer1, helper_1_layer2, helper_1_layer3, helper_J_LN, helper_β, helper_1overβ, helper_ϵ)

    predicted_activity = dists_to_networkstat_to_predicted_activity_gpu(NW1, NW2, NW3, M, networkstat_to_trials_multiplier, dists,
            helper_1_layer1, helper_1_layer2, helper_1_layer3)

    res = CUDA.sum( log_likelihood_Poisson_stable_gpu2(predicted_activity, activity, helper_J_LN, helper_β, helper_1overβ, helper_ϵ) .*
            activity_mask )

    return res
end




function maximum_likelihood_networkstat(dists::Array{Float64,2}, activity::Array{Float64,2}, activity_mask::Array{Float64,2},
        networkstat_to_trials_multiplier::Array{Float64,2}, num_networkstat::Integer, iteration::Integer=1000, γ::Float64=0.01)
        J = size(activity)[1]
        K = size(dists)[1]
        L = size(dists)[2]
        T_original = size(activity)[2]
        T = 16
        U = num_networkstat

        activity_copy = activity
        activity_mask_copy = activity_mask

        dists = dists |> gpu
        activity = activity |> gpu
        activity_mask = activity_mask |> gpu
        networkstat_to_trials_multiplier = networkstat_to_trials_multiplier |> gpu

        intermediate_layer_size = 64
        helper_1_layer1 = ones((intermediate_layer_size, L)) |> gpu
        helper_1_layer2 = ones((intermediate_layer_size, L)) |> gpu
        helper_1_layer3 = ones((num_networkstat, L)) |> gpu

        helper_J_T = ones((J, T)) |> gpu
        β = 100.0
        helper_β = ( β .* ones((J,T)) ) |> gpu
        helper_1overβ = ( (1.0/β) .* ones((J,T)) ) |> gpu

        helper_β2 = ( β .* ones((J,L)) ) |> gpu
        helper_1overβ2 = ( (1.0/β) .* ones((J,L)) ) |> gpu

        ϵ = 1.0e-6
        helper_ϵ = (ϵ .* ones((J, T))) |> gpu

        δ = 0.01
        helper_δ = ( δ .* ones((J, L)) ) |> gpu

        avg_number_terms = (1.0/sum(activity_mask_copy))
        avg_log_D = sum(log_factorial.(activity_copy) .* activity_mask_copy) * avg_number_terms

        γNW1 = γ
        γNW2 = γ
        γNW3 = γ
        γM = γ * 100

        rng = MersenneTwister(1)

        println("selecting candidates: ")
        #candidate_num = 30
        candidate_num = 5
        NW1_candidates = Array{Float64,3}(undef, (candidate_num, intermediate_layer_size, K))
        NW2_candidates = Array{Float64,3}(undef, (candidate_num, intermediate_layer_size, intermediate_layer_size))
        NW3_candidates = Array{Float64,3}(undef, (candidate_num, U, intermediate_layer_size))
        M_candidates = Array{Float64,3}(undef, (candidate_num, J, U))
        result_candidates = Array{Float64,1}(undef, candidate_num)

        for v in 1:candidate_num
                print("$(v) ")
                NW1_v = Array{Float64,2}(undef, (intermediate_layer_size, K))
                NW2_v = Array{Float64,2}(undef, (intermediate_layer_size, intermediate_layer_size))
                NW3_v = Array{Float64,2}(undef, (U, intermediate_layer_size))
                M_v = Array{Float64,2}(undef, (J, U))

                randn!(NW1_v)
                NW1_v = NW1_v ./ 10
                randn!(NW2_v)
                NW2_v = NW2_v ./ 10
                randn!(NW3_v)
                NW3_v = NW3_v ./ 10
                randn!(M_v)
                M_v = M_v ./10

                NW1_candidates[v,:,:] = NW1_v
                NW2_candidates[v,:,:] = NW2_v
                NW3_candidates[v,:,:] = NW3_v
                M_candidates[v,:,:] = M_v

                NW1_v = NW1_v |> gpu
                NW2_v = NW2_v |> gpu
                NW3_v = NW3_v |> gpu
                M_v = M_v |> gpu

                sampled_indices = sample(rng, 1:T_original, T, replace=false, ordered=true)
                print( (1.0/CUDA.sum(activity_mask[:,sampled_indices])) * log_likelihood_networkstat_gpu(
                        NW1_v, NW2_v, NW3_v, M_v, networkstat_to_trials_multiplier[:,sampled_indices], dists, activity[:,sampled_indices], activity_mask[:,sampled_indices],
                        helper_1_layer1, helper_1_layer2, helper_1_layer3, helper_J_T, helper_β, helper_1overβ, helper_ϵ) - avg_log_D, " ")

                for w in 1:3000
                #for w in 1:0
                        sampled_indices = sample(rng, 1:T_original, T, replace=false, ordered=true)
                        grad = gradient((NW1_v, NW2_v, NW3_v, M_v)->( (1.0/CUDA.sum(activity_mask[:,sampled_indices])) * log_likelihood_networkstat_gpu(
                                NW1_v, NW2_v, NW3_v, M_v, networkstat_to_trials_multiplier[:,sampled_indices], dists, activity[:,sampled_indices], activity_mask[:,sampled_indices],
                                helper_1_layer1, helper_1_layer2, helper_1_layer3, helper_J_T, helper_β, helper_1overβ, helper_ϵ) -
                                10.0 * larger_than_0_regularization_firing2(
                                M_v * input_to_networkstat_gpu(NW1_v, NW2_v, NW3_v, dists, helper_1_layer1, helper_1_layer2, helper_1_layer3),
                                helper_β2, helper_1overβ2, helper_δ)  -
                                0.001 * ( CUDA.sum(NW1_v .* NW1_v) + CUDA.sum(NW2_v .* NW2_v) +
                                CUDA.sum(NW3_v .* NW3_v)  ) ),
                                NW1_v, NW2_v, NW3_v, M_v)
                        NW1_v .+= (γNW1 .* (grad[1]))
                        NW2_v .+= (γNW2 .* (grad[2]))
                        NW3_v .+= (γNW3 .* (grad[3]))
                        M_v .+= (γM .* (grad[4]))
                        if w % 100 == 0
                                GC.gc()
                                #CuArrays.reclaim()
                        end
                end
                sampled_indices = sample(rng, 1:T_original, T, replace=false, ordered=true)
                result_candidates[v] = (1.0/CUDA.sum(activity_mask[:,sampled_indices])) * log_likelihood_networkstat_gpu(
                        NW1_v, NW2_v, NW3_v, M_v, networkstat_to_trials_multiplier[:,sampled_indices], dists, activity[:,sampled_indices], activity_mask[:,sampled_indices],
                        helper_1_layer1, helper_1_layer2, helper_1_layer3, helper_J_T, helper_β, helper_1overβ, helper_ϵ) - avg_log_D
                print(result_candidates[v]," ")

                #GC.gc()
                ##CuArrays.reclaim()
        end
        println("selected candidate preresult: ", result_candidates[argmax_not_NaN(result_candidates)])
        NW1 = NW1_candidates[argmax_not_NaN(result_candidates),:,:] |> gpu
        NW2 = NW2_candidates[argmax_not_NaN(result_candidates),:,:] |> gpu
        NW3 = NW3_candidates[argmax_not_NaN(result_candidates),:,:] |> gpu
        M = M_candidates[argmax_not_NaN(result_candidates),:,:] |> gpu

        GC.gc()
        #CuArrays.reclaim()


        for i in 1:iteration
                sampled_indices = sample(rng, 1:T_original, T, replace=false, ordered=true)
                grad = gradient((NW1, NW2, NW3, M)->( (1.0/CUDA.sum(activity_mask[:,sampled_indices])) * log_likelihood_networkstat_gpu(
                        NW1, NW2, NW3, M, networkstat_to_trials_multiplier[:,sampled_indices], dists, activity[:,sampled_indices], activity_mask[:,sampled_indices],
                        helper_1_layer1, helper_1_layer2, helper_1_layer3, helper_J_T, helper_β, helper_1overβ, helper_ϵ) -
                        10.0 * larger_than_0_regularization_firing2(
                        M * input_to_networkstat_gpu(NW1, NW2, NW3, dists, helper_1_layer1, helper_1_layer2, helper_1_layer3),
                        helper_β2, helper_1overβ2, helper_δ) -
                        0.001 * ( CUDA.sum(NW1 .* NW1) + CUDA.sum(NW2 .* NW2) +
                        CUDA.sum(NW3 .* NW3)  ) ),
                        NW1, NW2, NW3, M)
                NW1 .+= (γNW1 .* (grad[1]))
                NW2 .+= (γNW2 .* (grad[2]))
                NW3 .+= (γNW3 .* (grad[3]))
                M .+= (γM .* (grad[4]))


                if (i % 5000 == 0) || (i == 1)
                        sampled_indices = sample(rng, 1:T_original, T, replace=false, ordered=true)
                        current_res = (1.0/CUDA.sum(activity_mask[:,sampled_indices])) * log_likelihood_networkstat_gpu(
                                NW1, NW2, NW3, M, networkstat_to_trials_multiplier[:,sampled_indices], dists, activity[:,sampled_indices], activity_mask[:,sampled_indices],
                                helper_1_layer1, helper_1_layer2, helper_1_layer3, helper_J_T, helper_β, helper_1overβ, helper_ϵ) - avg_log_D
                        current_penalty1 = 10.0 * larger_than_0_regularization_firing2(
                                M * input_to_networkstat_gpu(NW1, NW2, NW3, dists, helper_1_layer1, helper_1_layer2, helper_1_layer3),
                                helper_β2, helper_1overβ2, helper_δ)
                        current_penalty2 = 0.001 * ( CUDA.sum(NW1 .* NW1) + CUDA.sum(NW2 .* NW2) +
                                CUDA.sum(NW3 .* NW3)  )
                        println("\n iteration ", i, "   ", current_res,"   ",current_penalty1, "   ", current_penalty2)
                        println(NW1[1, 1:20:end])
                        println(grad[1][1, 1:20:end])
                        println(NW2[1, 1:20:end])
                        println(grad[2][1, 1:20:end])
                        println(NW3[1, 1:20:end])
                        println(grad[3][1, 1:20:end])
                        println(M[10,:])
                        println(grad[4][10,:])
                        if isnan(current_res)
                                break
                        end
                end

                if i % 200 == 0
                        # explicitly perform gabarge collection
                        GC.gc()
                end
        end

        sampled_indices = sample(rng, 1:T_original, T, replace=false, ordered=true)
        result = (1.0/CUDA.sum(activity_mask[:,sampled_indices])) * log_likelihood_networkstat_gpu(
                NW1, NW2, NW3, M, networkstat_to_trials_multiplier[:,sampled_indices], dists, activity[:,sampled_indices], activity_mask[:,sampled_indices],
                helper_1_layer1, helper_1_layer2, helper_1_layer3, helper_J_T, helper_β, helper_1overβ, helper_ϵ) - avg_log_D

        GC.gc()
        # reclaim cached memory to avoid being out of memory CUDA error
        #CuArrays.reclaim()

        return (NW1 |> cpu), (NW2 |> cpu), (NW3 |> cpu), (M |> cpu), result

end






function input_to_network_softmax(NW1, NW2, NW3, input)
    K = size(NW3)[1]
    layer1_output = ELU_approximate.(NW1 * input)
    layer2_output = ELU_approximate.(NW2 * layer1_output)
    layer3_output = ELU_approximate.(NW3 * layer2_output)
    exp_layer3_output = exp.(layer3_output)
    softmax_output = exp_layer3_output ./ (ones((K,1)) * sum(exp_layer3_output, dims=1))
    return softmax_output
end

function input_to_network_softmax_gpu(NW1, NW2, NW3, input, helper_1_layer1, helper_1_layer2, helper_1_layer3,helper_1_K)
    layer1_output = ELU_approximate_gpu(NW1 * input, helper_1_layer1)
    layer2_output = ELU_approximate_gpu(NW2 * layer1_output, helper_1_layer2)
    layer3_output = ELU_approximate_gpu(NW3 * layer2_output, helper_1_layer3)
    exp_layer3_output = CUDA.exp.(layer3_output)
    softmax_output = exp_layer3_output ./ (helper_1_K * CUDA.transpose(helper_1_K) * exp_layer3_output)
    return softmax_output
end




function network_decode_CDFs(W, NW1, NW2, NW3, activity)
    U = size(W)[1]
    J = size(W)[2]
    T = size(activity)[2]
    K = size(NW3)[1]

    proj = W * activity
    decoded_probas = input_to_network_softmax(NW1, NW2, NW3, proj)

    Y = zeros((K-1, K))
    for i in 1:(K-1)
        for j in 1:i
            Y[i,j] = 1.0
        end
    end
    predicted_CDFs = Y * decoded_probas

    return predicted_CDFs
end

function network_decode_dists(W, NW1, NW2, NW3, activity)
    U = size(W)[1]
    J = size(W)[2]
    T = size(activity)[2]
    K = size(NW3)[1]

    proj = W * activity
    decoded_probas = input_to_network_softmax(NW1, NW2, NW3, proj)

    return decoded_probas
end


function network_decode_CDFs2(W, NW1, NW2, NW3, activity)
    U = size(W)[1]
    J = size(W)[2]
    T = size(activity)[2]
    K = size(NW3)[1]

    proj = W * activity
    decoded_probas = input_to_network_softmax(NW1, NW2, NW3, proj)

    Y = zeros((K, K))
    for i in 1:K
        for j in 1:i
            Y[i,j] = 1.0
        end
    end
    predicted_CDFs = Y * decoded_probas

    return predicted_CDFs
end


function network_decode_CDFs_gpu(W, NW1, NW2, NW3, activity, helper_1_layer1, helper_1_layer2, helper_1_layer3, helper_1_K, helper_accumulation)
    U = size(W)[1]
    J = size(W)[2]
    T = size(activity)[2]
    K = size(NW3)[1]

    proj = W * activity
    decoded_probas = input_to_network_softmax_gpu(NW1, NW2, NW3, proj, helper_1_layer1, helper_1_layer2, helper_1_layer3,helper_1_K)

    predicted_CDFs = helper_accumulation * decoded_probas

    return predicted_CDFs
end


function network_Wasserstein_distance_CDFs_weighted(W, NW1, NW2, NW3, activity, reward_values, target_CDFs, trial_nums)
    U = size(W)[1]
    J = size(W)[2]
    T = size(activity)[2]
    K = length(reward_values)

    predicted_CDFs = network_decode_CDFs(W, NW1, NW2, NW3, activity)

    reward_values_diff = reward_values[2:end] .- reward_values[1:(end-1)]
    diff = abs.(predicted_CDFs .- target_CDFs)
    # return mean(transpose(reward_values_diff) * diff)
    per_trial_wass = transpose(reward_values_diff) * diff
    cum_trial_nums = cumsum(trial_nums)
    prepend!(cum_trial_nums, [0])

    per_type_avg_wass = [mean(per_trial_wass[cum_trial_nums[i]+1:cum_trial_nums[i+1]]) for i in 1:length(trial_nums)]
    return mean(per_type_avg_wass)
end


function network_Wasserstein_distance_CDFs(W, NW1, NW2, NW3, activity, reward_values, target_CDFs)
    U = size(W)[1]
    J = size(W)[2]
    T = size(activity)[2]
    K = length(reward_values)

    predicted_CDFs = network_decode_CDFs(W, NW1, NW2, NW3, activity)

    reward_values_diff = reward_values[2:end] .- reward_values[1:(end-1)]
    diff = abs.(predicted_CDFs .- target_CDFs)
    return mean(transpose(reward_values_diff) * diff)
end


function network_Wasserstein_distance_CDFs_gpu(W, NW1, NW2, NW3, activity, reward_values_diff, target_CDFs,
        helper_1_layer1, helper_1_layer2, helper_1_layer3, helper_1_K, helper_accumulation)

    predicted_CDFs = network_decode_CDFs_gpu(W, NW1, NW2, NW3, activity, helper_1_layer1, helper_1_layer2, helper_1_layer3, helper_1_K, helper_accumulation)

    diff = CUDA.abs.(predicted_CDFs .- target_CDFs)
    return CUDA.sum(CUDA.transpose(reward_values_diff) * diff)
end




function minimize_Wasserstein_network(activity, reward_values, target_CDFs, num_networkstat, iteration=1000, γ=0.01, λ=1.0)
        J = size(activity)[1]
        T = size(activity)[2]
        println(J)
        if J < 1500
            div = 10
        elseif J > 2000
            div = 20
        else
            div = 15
        end
        K = length(reward_values)
        U = num_networkstat

        activity = activity |> gpu

        reward_values_diff = reward_values[2:end] .- reward_values[1:(end-1)]
        if length(reward_values_diff) > 1
            reward_values_diff = reshape(reward_values_diff, (length(reward_values_diff),1))
        else
            reward_values_diff = ones((1,1)) * reward_values_diff[1]
        end
        reward_values_diff = reward_values_diff |> gpu

        target_CDFs = target_CDFs |> gpu

        #intermediate_layer_size = 16
        intermediate_layer_size = 32
        helper_1_layer1 = ones((intermediate_layer_size, T)) |> gpu
        helper_1_layer2 = ones((intermediate_layer_size, T)) |> gpu
        helper_1_layer3 = ones((K, T)) |> gpu

        helper_1_K = ones((K,1))
        helper_1_K = helper_1_K |> gpu

        helper_accumulation = zeros((K-1, K))
        for k1 in 1:(K-1)
            for k2 in 1:k1
                helper_accumulation[k1, k2] = 1.0
            end
        end
        helper_accumulation = helper_accumulation |> gpu


        γW = γ * 1
        γNW1 = γ
        γNW2 = γ
        γNW3 = γ


        avg_number_terms = (1.0/T)


        println("selecting candidates: ")
        candidate_num = 5
        W_candidates = Array{Float64,3}(undef, (candidate_num, U, J))
        NW1_candidates = Array{Float64,3}(undef, (candidate_num, intermediate_layer_size, U))
        NW2_candidates = Array{Float64,3}(undef, (candidate_num, intermediate_layer_size, intermediate_layer_size))
        NW3_candidates = Array{Float64,3}(undef, (candidate_num, K, intermediate_layer_size))
        result_candidates = Array{Float64,1}(undef, candidate_num)

        for v in 1:candidate_num

                W_v = Array{Float64,2}(undef, (U, J))
                NW1_v = Array{Float64,2}(undef, (intermediate_layer_size, U))
                NW2_v = Array{Float64,2}(undef, (intermediate_layer_size, intermediate_layer_size))
                NW3_v = Array{Float64,2}(undef, (K, intermediate_layer_size))

                randn!(W_v)
                W_v = W_v ./= div  # sqrt(J)
                randn!(NW1_v)
                NW1_v = NW1_v ./ div  # sqrt(J)
                randn!(NW2_v)
                NW2_v = NW2_v ./ div # sqrt(J)
                randn!(NW3_v)
                NW3_v = NW3_v ./ div # sqrt(J)

                #=
                W_candidates[v,:,:] = W_v
                NW1_candidates[v,:,:] = NW1_v
                NW2_candidates[v,:,:] = NW2_v
                NW3_candidates[v,:,:] = NW3_v
                =#

                W_v = W_v |> gpu
                NW1_v = NW1_v |> gpu
                NW2_v = NW2_v |> gpu
                NW3_v = NW3_v |> gpu

                if v % 1 == 0
                        print("$(v) ")
                        print(avg_number_terms *
                            network_Wasserstein_distance_CDFs_gpu(W_v, NW1_v, NW2_v, NW3_v, activity, reward_values_diff, target_CDFs,
                                helper_1_layer1, helper_1_layer2, helper_1_layer3, helper_1_K, helper_accumulation) +
                            0*(weight_penalty_regularization(NW1_v) + weight_penalty_regularization(NW2_v) + weight_penalty_regularization(NW3_v)) )
                        print(" ")
                end


                for w in 1:2000
                        grad = gradient((W_v, NW1_v, NW2_v, NW3_v)->( avg_number_terms *
                                network_Wasserstein_distance_CDFs_gpu(W_v, NW1_v, NW2_v, NW3_v, activity, reward_values_diff, target_CDFs,
                                    helper_1_layer1, helper_1_layer2, helper_1_layer3, helper_1_K, helper_accumulation) +
                                λ*(weight_penalty_regularization(NW1_v) + weight_penalty_regularization(NW2_v) + weight_penalty_regularization(NW3_v)) ),
                                W_v, NW1_v, NW2_v, NW3_v)
                        W_v .-= (γW .* (grad[1]))
                        NW1_v .-= (γNW1 .* (grad[2]))
                        NW2_v .-= (γNW2 .* (grad[3]))
                        NW3_v .-= (γNW3 .* (grad[4]))
                        if w % 100 == 0
                                GC.gc()
                                #CuArrays.reclaim()
                        end
                end


                W_candidates[v,:,:] = W_v |> cpu
                NW1_candidates[v,:,:] = NW1_v |> cpu
                NW2_candidates[v,:,:] = NW2_v |> cpu
                NW3_candidates[v,:,:] = NW3_v |> cpu


                result_candidates[v] = avg_number_terms *
                    network_Wasserstein_distance_CDFs_gpu(W_v, NW1_v, NW2_v, NW3_v, activity, reward_values_diff, target_CDFs,
                        helper_1_layer1, helper_1_layer2, helper_1_layer3, helper_1_K, helper_accumulation) +
                    0*(weight_penalty_regularization(NW1_v) + weight_penalty_regularization(NW2_v) + weight_penalty_regularization(NW3_v))

                if v % 1 == 0
                        print(result_candidates[v]," ")
                end

        end

        argmax_idx = argmin_not_NaN(result_candidates)
        println("selected candidate preresult: ",result_candidates[argmax_idx])
        W = W_candidates[argmax_idx,:,:] |> gpu
        NW1 = NW1_candidates[argmax_idx,:,:] |> gpu
        NW2 = NW2_candidates[argmax_idx,:,:] |> gpu
        NW3 = NW3_candidates[argmax_idx,:,:] |> gpu

        GC.gc()
        #CuArrays.reclaim()



        for i in 1:iteration
                #println("iteration ",i)
                grad = gradient((W, NW1, NW2, NW3)->( avg_number_terms *
                        network_Wasserstein_distance_CDFs_gpu(W, NW1, NW2, NW3, activity, reward_values_diff, target_CDFs,
                            helper_1_layer1, helper_1_layer2, helper_1_layer3, helper_1_K, helper_accumulation) +
                        λ*(weight_penalty_regularization(NW1) + weight_penalty_regularization(NW2) + weight_penalty_regularization(NW3)) ),
                        W, NW1, NW2, NW3)
                W .-= (γW .* (grad[1]))
                NW1 .-= (γNW1 .* (grad[2]))
                NW2 .-= (γNW2 .* (grad[3]))
                NW3 .-= (γNW3 .* (grad[4]))
                if (i % 10000 == 0) || (i == 1)
                        current_res1 = avg_number_terms * network_Wasserstein_distance_CDFs_gpu(W, NW1, NW2, NW3, activity, reward_values_diff, target_CDFs,
                                    helper_1_layer1, helper_1_layer2, helper_1_layer3, helper_1_K, helper_accumulation)
                        current_res2 = λ*(weight_penalty_regularization(NW1) + weight_penalty_regularization(NW2) + weight_penalty_regularization(NW3))
                        current_res = current_res1 + current_res2
                        println("\n iteration ", i, "   ", current_res1,"   ", current_res2)
                        println(W[1,1:4])
                        println(grad[1][1,1:4])
                        println(NW1[1,1:1])
                        println(grad[2][1,1:1])
                        println(NW2[1,1:6])
                        println(grad[3][1,1:6])
                        println(NW3[1,1:2])
                        println(grad[4][1,1:2])
                        if isnan(current_res)
                                break
                        end
                end

                if i % 200 == 0
                        # explicitly perform gabarge collection
                        GC.gc()
                end

        end

        GC.gc()
        # reclaim cached memory to avoid being out of memory CUDA error
        #CuArrays.reclaim()

        return (W |> cpu), (NW1 |> cpu), (NW2 |> cpu), (NW3 |> cpu)

end


function Wasserstein_distance_given_CDF(CDF_without_one, reward_values, target_CDFs)
    K = size(target_CDFs)[1] + 1
    T = size(target_CDFs)[2]

    diff = zeros((K-1, T))
    for t in 1:T
        diff[:,t] = CDF_without_one .- target_CDFs[:,t]
    end
    diff = abs.(diff)

    reward_values_diff = reward_values[2:end] .- reward_values[1:(end-1)]
    return mean(transpose(reward_values_diff) * diff)
end


function CDFs_to_ps(CDFs)
    K = size(CDFs)[1]
    T = size(CDFs)[2]

    res = zeros((K, T))
    for t in 1:T
        res[1,t] = CDFs[1,t]
        for k in 2:K
            res[k,t] = CDFs[k,t] - CDFs[k-1,t]
        end
    end
    return res
end

function validity_rate_and_CDFs(CDFs2, reward_values)
    K = size(CDFs2)[1]
    T = size(CDFs2)[2]

    CDFs_x = zeros((2*K,T))
    CDFs_y = zeros((2*K,T))

    global count = 0
    for t in 1:T
        if (CDFs2[1,t] >= 0) && (minimum(CDFs2[2:end,t] .- CDFs2[1:(end-1),t]) >= 0)
            global count += 1
        end

        CDFs_x[1, t] = reward_values[1]
        CDFs_y[1, t] = 0.0
        CDFs_x[2, t] = reward_values[1]
        CDFs_y[2, t] = CDFs2[1, t]
        for k in 2:K
            CDFs_x[2*k-1, t] = reward_values[k]
            CDFs_y[2*k-1, t] = CDFs2[k-1,t]
            CDFs_x[2*k, t] = reward_values[k]
            CDFs_y[2*k, t] = CDFs2[k, t]
        end

    end
    return count/T, CDFs_x, CDFs_y
end


# function plot_CDF(CDF_x, CDF_y, reward_values, color, file)
#     K = length(reward_values)
#     #println(size(CDF_x))
#     #println(size(CDF_y))
#     clf()
#     plot(CDF_x[1,:], CDF_y[1,:], color=color, linewidth=1.5, linestyle="-", alpha=0.9)
#     xticks(reward_values, fontsize=8)
#     yticks([0,1],fontsize=8)

#     tight_layout()
#     savefig(file, format="pdf", bbox_inches="tight")
# end


# function plot_decoded_CDFs(CDFs_x, CDFs_y, dist, gt_CDFs_x, gt_CDFs_y, ref_CDF_x, ref_CDF_y,
#         distributions, reward_values, colors, file)
#     T = size(dist)[2]
#     U = size(distributions)[1]
#     K = size(distributions)[2]
#     clf()
#     for u in 1:U
#         subplot(Int(ceil(U/2)), 2, u)
#         for t in 1:T
#             if dist[:,t] == distributions[u,:]
#                 plot(CDFs_x[:,t], CDFs_y[:,t], color=colors[u], linewidth=0.5, linestyle="-", alpha=0.4)
#             end
#         end
#         plot(gt_CDFs_x[u,:], gt_CDFs_y[u,:], color="grey", linewidth=1.5, linestyle="--", alpha=0.6)
#         plot(ref_CDF_x[1,:], ref_CDF_y[1,:], color="saddlebrown", linewidth=0.6, linestyle="-.", alpha=0.99)
#         xticks(reward_values, fontsize=8)
#         yticks([0,1],fontsize=8)
#     end
#     tight_layout()
#     savefig(file, format="pdf", bbox_inches="tight")
# end

# function plot_decoded_valid_distributions_quantile_code(quantiles, dist, distributions, reward_values, constant_intervals, colors, file)
#     T = size(dist)[2]
#     U = size(distributions)[1]
#     K = size(distributions)[2]
#     clf()
#     for u in 1:U
#         subplot(Int(ceil(U/2)), 2, u)
#         for t in 1:T
#             if (dist[:,t] == distributions[u,:]) && (minimum(quantiles[2:end,t] .- quantiles[1:(end-1),t]) >= 0)
#                 plot(quantiles[:,t], constant_intervals[2:end] .- constant_intervals[1:(end-1)], color=colors[u], linewidth=0.5, linestyle="-", alpha=0.4)
#             end
#         end
#         plot(reward_values, distributions[u,:], color="grey", linewidth=1.5, linestyle="--", alpha=0.6)
#         xticks(reward_values, fontsize=8)
#         yticks([0,1],fontsize=8)
#     end
#     tight_layout()
#     savefig(file, format="pdf", bbox_inches="tight")

# end


# function plot_decoded_valid_distributions(predicted_distributions, dist, distributions, reward_values, colors, file)
#     T = size(dist)[2]
#     U = size(distributions)[1]
#     K = size(distributions)[2]
#     clf()
#     for u in 1:U
#         subplot(Int(ceil(U/2)), 2, u)
#         for t in 1:T
#             if (dist[:,t] == distributions[u,:]) && (minimum(predicted_distributions[:,t]) >= 0)
#                 plot(reward_values, predicted_distributions[:,t], color=colors[u], linewidth=0.5, linestyle="-", alpha=0.4)
#             end
#         end
#         plot(reward_values, distributions[u,:], color="grey", linewidth=1.2, linestyle="--", alpha=1.0)
#         xticks(reward_values, fontsize=8)
#         yticks([0,1],fontsize=8)
#         if u % 2 == 1
#             ylabel("proba",fontsize=8)
#         end
#         if u in [U-1,U]
#             xlabel("reward",fontsize=8)
#         end
#     end
#     tight_layout()
#     savefig(file, format="pdf", bbox_inches="tight")

# end


function expectile_loss(e, α::Float64, p::Array{Float64,1}, rewards::Array{Float64,1})
        w = Array([r<=e ? (1-α) : α for r in rewards])
        return sum(p .* w .* ((rewards .- e).^2))
end

function numerical_optimization_expectile(α::Float64, p::Array{Float64,1}, rewards::Array{Float64,1},
        iteration::Integer=1000, γ::Float64=0.01)

        e = rewards[argmax(p)]
        #e = (rewards[1] + rewards[end])/2.0
        for i in 1:iteration
                grad = gradient(e->expectile_loss(e, α, p, rewards), e)[1]
                γ_i = γ*(1/(1+0.001*i))
                e -= ((γ_i)*grad)
        end
        return e
end

function weight_penalty_regularization(W)
    return CUDA.sum(W .^2)
end


function occurrence_count(a)
    components = unique(a)
    u = []
    for x in components
        x_count = 0
        for v in a
            if v == x
                x_count += 1
            end
        end
        push!(u, x_count)
    end
    u = convert(Array{Int64,1}, u)
    return components, u, [x/sum(u) for x in u]
end




# function plot_pcolormesh(data, file_name, cmap_name, xtick_list, ytick_list, xlabel_name, ylabel_name)
#         clf()
#         #c = pcolormesh(data, cmap = "gnuplot2",vmin = -180.0, vmax=180.0)
#         #c = pcolormesh(data, cmap = "Oranges",vmin = 20.0, vmax=50.0)
#         #c = pcolormesh(data, cmap = "PRGn")
#         #c = pcolormesh(data, cmap = "PRGn",vmin = -1000.0, vmax=2200.0)
#         #c = pcolormesh(data, cmap = "Greens", vmin = 1.5, vmax=3.0)
#         #c = pcolormesh(data, cmap = "Oranges",vmin = 1.5, vmax=3.0)
#         if cmap_name == "gnuplot2"
#                 #c = pcolormesh(data, cmap = "gnuplot2")
#                 c = pcolormesh(data, cmap = "gnuplot2",vmin = -0.5, vmax=1.0)
#                 #c = pcolormesh(data, cmap = "gnuplot2",vmin = -0.1, vmax=0.4)
#         elseif cmap_name == "gist_heat"
#                 #c = pcolormesh(data, cmap = "gist_heat")
#                 c = pcolormesh(data, cmap = "gist_heat",vmin = -0.8, vmax=0.2)
#                 #c = pcolormesh(data, cmap = "gist_heat")
#         elseif cmap_name == "terrain"
#                 #c = pcolormesh(data, cmap = "terrain")
#                 c = pcolormesh(data, cmap = "terrain",vmin = 0.0, vmax= 0.5 )
#         elseif cmap_name == "bwr"
#                 #c = pcolormesh(data, cmap = "bwr")
#                 c = pcolormesh(data, cmap = "bwr",vmin = -1.0, vmax= 1.0 )
#         end
#         #c = pcolormesh(data, cmap = "gnuplot2")
#         colorbar(c)
#         yticks([t - 0.5 for t in 1:length(ytick_list)], ytick_list, fontsize=8)
#         #yticks([t for t in 1:20],[5 * t for t in 1:20],fontsize=8)
#         #yticks([t for t in 1:20],[1 * t for t in 1:20],fontsize=8)

#         ylabel(ylabel_name,fontsize=10)
#         #ylabel("β of tuning (tuning sharpness)",fontsize=10)
#         #ylabel("κ of tuning (tuning concentration)",fontsize=10)

#         xticks([t - 0.5 for t in 1:length(xtick_list)], [x for x in xtick_list], fontsize=8,rotation=30)

#         xlabel(xlabel_name,fontsize=10)
#         tight_layout()
#         savefig(file_name, format="pdf", bbox_inches="tight")
#         clf()

# end

function setup_decoding(same_data, d, time_window_idx, dist_rng)
    data = [same_data][d]

    resp_data = data["cue_spk_cnts"][:,:,:,time_window_idx]
    println(size(resp_data))
    session_id_data = data["neuron_info"]["session_ids"]
    #println(size(session_id_data))
    mouse_ids = sort(unique(data["neuron_info"]["mouse_ids"]))
    #println(mouse_ids)
    session_ids = sort(unique(data["neuron_info"]["session_ids"]))
    #println(session_ids)
    distributions = ground_truth_distributions(data["protocol_info"]["dists"])
    # get rid of the last distribution, corresponding to unexpected trials
    distributions = distributions[dist_rng,:]
    println("distributions: ", distributions)
    println("distribution size: ", size(distributions))
    reward_values = ground_truth_reward_values(data["protocol_info"]["dists"])
    reward_values = convert(Array{Float64,1}, reward_values)
    println(reward_values)

    CDFs = CDFs_given_distributions(distributions)
    #println(size(CDFs))
    CDFs_without_one = CDFs_given_distributions_without_one(distributions)
    #println(size(CDFs_without_one))
    constant_intervals = constant_inverse_CDF_intervals(CDFs)
    #println(constant_intervals)
    quantiles_on_constant_intervals = quantiles_on_constant_inverse_CDF_intervals(CDFs, constant_intervals, reward_values)
    #println(quantiles_on_constant_intervals)

    regions = data["neuron_info"]["str_regions"]
    regions = [typeof(x) == String ? convert(String,x) : "nothing" for x in regions]

    dorso_ventral_coordinates = data["neuron_info"]["depths"]

    return data, resp_data, session_id_data, mouse_ids, session_ids, distributions, reward_values, CDFs, CDFs_without_one, constant_intervals, 
        quantiles_on_constant_intervals, regions, dorso_ventral_coordinates
    
end


function get_mouse_session_dict(data)

    mouse_ids = sort(unique(data["neuron_info"]["mouse_ids"]))
    # mouse_names = sort(unique(data["neuron_info"]["names"]))
    # mouse_id_name_pairs = unique([[data["neuron_info"]["mouse_ids"][i], data["neuron_info"]["names"][i]] for i in 1:length(data["neuron_info"]["mouse_ids"])])
    # mouse_id_name_map = Dict()
    # for i in 1:length(mouse_id_name_pairs)
    #     mouse_id_name_map[mouse_id_name_pairs[i][1]] = mouse_id_name_pairs[i][2]
    # end
    
    #println(mouse_id_name_map)
    session_mouse_pairs = unique([(data["neuron_info"]["session_ids"][i], data["neuron_info"]["mouse_ids"][i])
        for i in 1:length(data["neuron_info"]["session_ids"])])
    mouse_session_dict = Dict()
    for mouse in mouse_ids
        mouse_session_dict[mouse] = []
    end
    for session_mouse in session_mouse_pairs
        session = session_mouse[1]
        mouse = session_mouse[2]
        append!(mouse_session_dict[mouse], [session])
    end
    for mouse in mouse_ids
        mouse_session_dict[mouse] = sort(mouse_session_dict[mouse]; alg=QuickSort)
    end
    #println(mouse_session_dict)

    return mouse_session_dict
end


function setup_riskindex(all_activity, test_activity, train_means, train_maxs, selected_risk_indices, selection_indices; fraction=0.25)

    valid_selected_risk_indices = [x for x in selected_risk_indices if !isnan(x)]
    #println(length(valid_selected_risk_indices))
    valid_selected_risk_indices = sort(valid_selected_risk_indices; alg=QuickSort)
    for i=1:2
        if i == 1
            risk_index_threshold = valid_selected_risk_indices[Int(round(length(valid_selected_risk_indices) * (1-fraction)))]
            dropout_indices = [i for i in 1:length(selection_indices) if (!isnan(selected_risk_indices[i])) && (selected_risk_indices[i] > risk_index_threshold)]
        else
            risk_index_threshold = valid_selected_risk_indices[Int(round(length(valid_selected_risk_indices) * fraction))]
            dropout_indices = [i for i in 1:length(selection_indices) if (!isnan(selected_risk_indices[i])) && (selected_risk_indices[i] <= risk_index_threshold)]
        end

        println(length(dropout_indices), dropout_indices)

        dropped_out = copy(test_activity)
        dropped_out[dropout_indices, :] = zeros((length(dropout_indices), size(test_activity)[2]))
        push!(all_activity, dropped_out)

        # Set quartiles of dorso-ventral coordinate to mean across trial types
        set_dv_to_mean = copy(test_activity)
        set_dv_to_mean[dropout_indices, :] = train_means[dropout_indices, :]
        push!(all_activity, set_dv_to_mean)

        # Set quartiles of dorso-ventral coordinate to max across trial types
        set_dv_to_max = copy(test_activity)
        set_dv_to_max[dropout_indices, :] = train_maxs[dropout_indices, :]
        push!(all_activity, set_dv_to_max)

    end

    return all_activity
    
end


function get_dist_neuron_labels(data)
    dist_neurons_dict = Dict()
    for k in keys(data["dist_neurons"])
        dist_neurons_dict[convert(Int, k)] = k
    end
    dist_neuron_labels = [Bool(x) for x in data["dist_neurons"][dist_neurons_dict[95]]["all"]]
    return dist_neuron_labels
end


function push_dropouts(all_activity, test_activity, dropout_indices, train_means, train_maxs)
    dropped_out = copy(test_activity)
    dropped_out[dropout_indices, :] = zeros((length(dropout_indices), size(test_activity)[2]))
    push!(all_activity, dropped_out)

    # Set quartiles of dorso-ventral coordinate to mean across trial types
    set_dv_to_mean = copy(test_activity)
    set_dv_to_mean[dropout_indices, :] = train_means[dropout_indices, :]
    push!(all_activity, set_dv_to_mean)

    # Set quartiles of dorso-ventral coordinate to average max across trial types
    set_dv_to_max = copy(test_activity)
    set_dv_to_max[dropout_indices, :] = train_maxs[dropout_indices, :]
    push!(all_activity, set_dv_to_max)

    return all_activity
end

function setup_dv(all_activity, test_activity, train_means, train_maxs, n_dv_splits, dvs_session, n_neurons, seed)

    # test_activity, train_activity, shuffle_indicator, train_dist, test_dist, all_test_indices, all_train_indices = setup_session(
    #     i, data_session, trial_numbers, distributions, repetition_num, t, d, session_id)

    # all_activity = [test_activity]
    # test_activity = all_activity[1]

    # if all_activity == "skip"
    #     return "skip"
    # end

    # XXX: ADAM
    for v in 1:(n_dv_splits+1)
        if v <= n_dv_splits
            sorted_selected_dorso_ventral_coordinates = sort(dvs_session; alg=QuickSort)
            coordinate_threshold1 = sorted_selected_dorso_ventral_coordinates[Int(round(n_neurons * ((v-1)/n_dv_splits))) + 1]
            coordinate_threshold2 = sorted_selected_dorso_ventral_coordinates[Int(round(n_neurons * (v/n_dv_splits)))]
            dropout_indices = [ind for ind in 1:n_neurons if
                (coordinate_threshold1 <= dvs_session[ind]) && (dvs_session[ind] <= coordinate_threshold2)]
            #println(length(dropout_indices), dropout_indices)
        else
            dropout_indices = randperm(MersenneTwister(seed), n_neurons)[1:n_dv_splits:end]
            # println(dropout_indices)
        end
        all_activity = push_dropouts(all_activity, test_activity, dropout_indices, train_means, train_maxs)
    end

    return all_activity

end


# function push_selected(all_activity, dropout_bool)
#     dropout_indices = findall(x -> x>0, dropout_bool)
#     if length(dropout_indices) > fraction * n_neurons
#         dropout_indices = sample(rng, dropout_indices, round(Int, fraction * n_neurons), replace=false)
#         all_activity = push_dropouts(all_activity, test_activity, dropout_indices, train_means, train_maxs)
#     else  # if either dorsal or ventral doesn't work, the whole session will be skipped
#         for j in 1:3  # push_dropouts pushes three things, so keep length the same
#             push!(all_activity, fill(NaN, size(test_activity)))
#         end
#         # println("skip")
#         # return "skip"
#     end
#     return all_activity


function setup_split_bin(all_activity, test_activity, train_means, train_maxs, reward_values, bins_session, n_neurons, seed; fraction=.25)
    
    # if all_activity == "skip"
    #     return "skip"
    # end

    # TODO: DON'T HARD CODE BINS. USE reward_values
    rng = MersenneTwister(seed)
    for v in 1:2
        if v == 1
            dropout_bool = in.(bins_session, Ref([0, 2]))
        elseif v == 2
            dropout_bool = in.(bins_session, Ref([4, 6]))
        end

        dropout_indices = findall(x -> x>0, dropout_bool)
        if length(dropout_indices) > fraction * n_neurons
            dropout_indices = sample(rng, dropout_indices, round(Int, fraction * n_neurons), replace=false)
            all_activity = push_dropouts(all_activity, test_activity, dropout_indices, train_means, train_maxs)
        else  # if either low or high bins don't work, the whole session will be skipped
            for j in 1:3  # push_dropouts pushes three things, so keep length the same
                push!(all_activity, fill(NaN, size(test_activity)))
            end
            # println("skip")
            # return "skip"
        end
        
    end

    return all_activity

end



function setup_direction(all_activity, test_activity, train_means, train_maxs, reward_values, directions, n_neurons, seed; fraction=.25)
    

    rng = MersenneTwister(seed)
    for v in 1:2
        if v == 1
            dropout_bool = directions .== "Negative"
        elseif v == 2
            dropout_bool = directions .== "Positive"
        end

        dropout_indices = findall(x -> x>0, dropout_bool)
        if length(dropout_indices) > fraction * n_neurons
            dropout_indices = sample(rng, dropout_indices, round(Int, fraction * n_neurons), replace=false)
            all_activity = push_dropouts(all_activity, test_activity, dropout_indices, train_means, train_maxs)
        else  # if either Positive or Negative doesn't work, the whole session will be skipped
            for j in 1:3  # push_dropouts pushes three things, so keep length the same
                push!(all_activity, fill(NaN, size(test_activity)))
            end
        end
    end

    return all_activity

end


function setup_dv_region(all_activity, test_activity, train_means, train_maxs, n_dv_splits, regions_session, n_neurons, seed; fraction=.25)

    # if all_activity == "skip"
    #     return "skip"
    # end

    # XXX: ADAM
    rng = MersenneTwister(seed)
    for v in 1:n_dv_splits
        if v == 1
            dropout_bool = in.(regions_session, Ref(["lAcbSh", "VP", "core"]))
        elseif v == 2
            dropout_bool = in.(regions_session, Ref(["VMS", "VLS", "DMS", "DLS"]))
        elseif v == 3  # exclude VLS
            dropout_bool = in.(regions_session, Ref(["VMS", "DMS", "DLS"]))
        # else
        #     dropout_bool = ones(Bool, n_neurons)
        #     # dropout_neurons = randperm(rng, n_neurons)[1:n_dv_splits:end]
        #     # dropout_indices = zeros(Bool, n_neurons)
        #     # dropout_indices[dropout_neurons] .= 1
        end
        # println(dropout_indices)

        dropout_indices = findall(x -> x>0, dropout_bool)
        if length(dropout_indices) > fraction * n_neurons
            dropout_indices = sample(rng, dropout_indices, round(Int, fraction * n_neurons), replace=false)
            all_activity = push_dropouts(all_activity, test_activity, dropout_indices, train_means, train_maxs)
        else  # if either dorsal or ventral doesn't work, the whole session will be skipped
            # return "skip"
            for j in 1:3  # push_dropouts pushes three things, so keep length the same
                push!(all_activity, fill(NaN, size(test_activity)))
            end
        end
    end

    return all_activity

end


function sample_trial_indices(pseudo_set_idx, trial_num, randperm_indices, set_selected_num, nsplits, num_pseudotrial_per_dist, rng; replace=true)
    randperm_set_indices = randperm_indices[((pseudo_set_idx-1)*set_selected_num+1):(pseudo_set_idx*set_selected_num)]
    if pseudo_set_idx == nsplits
        randperm_set_indices = vcat(randperm_set_indices, randperm_indices[(pseudo_set_idx*set_selected_num+1):trial_num])
    end
    # println(randperm_set_indices)
    selected_trial_indices = sample(rng, randperm_set_indices, num_pseudotrial_per_dist, replace=replace, ordered=false)
    return selected_trial_indices
end


function setup_riskneuron(neuron_selection_idx, dist_neuron_labels, risk_indices)

    if neuron_selection_idx == 1
        neuron_selection = "distNeuron"
        selection_indices = [u for u in 1:length(dist_neuron_labels) if dist_neuron_labels[u]]
    elseif neuron_selection_idx == 2
        neuron_selection = "distNeuronValidRiskIdx"
        selection_indices = [u for u in 1:length(dist_neuron_labels) if (dist_neuron_labels[u]) && (!isnan(risk_indices[u]))]
    end
    println(neuron_selection)
    println(length(selection_indices))

    return neuron_selection, selection_indices

end


function setup_transfer(data_session, trial_numbers, distributions, standard_subset, standard_test, set, all_counts)

    neuron_num = size(data_session)[2]
    reward_value_num = size(distributions)[2]

    train_activity = zeros((neuron_num, 0))
    test_activity = zeros((neuron_num, 0))

    train_dist = zeros((reward_value_num, 0))
    test_dist = zeros((reward_value_num, 0))
    standard_test_dist = zeros((reward_value_num, 0))

    # all_test_indices = []
    # all_train_indices = []

    train_count = 0
    test_count = 0

    for t in 1:length(trial_numbers)  # n_dists
        trial_num = trial_numbers[t]
        if t in set
            train_indices = collect(1:trial_num)
            train_count += 1
            
            # # the part below applies if you only want to use each trial once per training set for either matched or mismatched.
            # # However, since we're averaging across sets for each session, we can use all trials each time and get a better estimate 
            # # of the true effect

            # if t in [1, 2]
            #     nsplits = div(n_transfers, 2)
            #     set_count = mod(i_set-1, nsplits) + 1
            # else
            #     nsplits = 2
            #     set_count = mod(all_counts[t], nsplits) + 1
            # end
            # all_counts[t] += 1
            # train_indices = collect(set_count:nsplits:trial_num) # sample_trial_indices(set_count, trial_num, randperm_indices_all[t], set_selected_num, nsplits, trial_num, rng; replace=false)
            # push!(all_train_indices, train_indices)
            # println(t)
            # println(train_indices)

            train_activity = hcat(train_activity, data_session[t,:, train_indices])
            train_dist = hcat(train_dist, distributions[standard_subset[train_count],:] * transpose(ones((size(train_indices)[1]))))
        else
            test_indices = collect(1:trial_num)
            test_count += 1
            # push!(all_test_indices, test_indices)
            test_activity = hcat(test_activity, data_session[t,:, test_indices])
            test_dist = hcat(test_dist, distributions[t,:] * transpose(ones((size(test_indices)[1]))))  # always test on correct dist
            standard_test_dist = hcat(standard_test_dist, distributions[standard_test[test_count],:] * transpose(ones((size(test_indices)[1]))))
        end
    end

    println(size(train_activity))
    println(size(test_activity))

    return train_activity, test_activity, train_dist, test_dist, standard_test_dist, all_counts
end


function setup_session(i, data_session, trial_numbers, distributions, repetition_num, t, d, session_id)  #; start=1, stop=nothing)

    # if stop === nothing
    #     stop = size(distributions)[1]
    # end

    # n_dists_to_use = stop - start + 1
    n_dists_to_use = length(trial_numbers)

    if i == 1
        shuffle_indicator = "matched"
        shuffled_indices = [x for x in 1:n_dists_to_use]
        # train_activity, test_activity, train_dist, test_dist, train_means, all_test_indices, all_train_indices = partition_train_test_data(data_session[start:stop, :, :], 
        #     trial_numbers[start:stop], distributions[start:stop, :], repetition_num, t, shuffled_indices)
        train_activity, test_activity, train_dist, test_dist, train_means, train_maxs, all_test_indices, all_train_indices = partition_train_test_data(
            data_session, trial_numbers, distributions, repetition_num, t, shuffled_indices)

    elseif i == 2
        shuffle_indicator = "shuffled"
        rng = MersenneTwister(t*1000 + d*100 + session_id)
        shuffled_indices = [x for x in 1:n_dists_to_use]
        while shuffled_indices == [x for x in 1:n_dists_to_use]
            shuffled_indices = shuffle(rng, shuffled_indices)
        end
        train_activity, test_activity, train_dist, test_dist, train_means, train_maxs, all_test_indices, all_train_indices = 
            partition_train_test_data(data_session, trial_numbers, distributions, repetition_num, t, shuffled_indices)
    end

    # println(size(train_means))  # which should be size n_neurons x n_test, not x n_train
    # println(train_means)

    println(size(train_activity))
    println(size(test_activity))

    # println(shuffled_indices)
    println(session_id, " ", shuffle_indicator)

    # println(dvs_session)
    # println(length(selection_indices))

    return test_activity, train_activity, shuffle_indicator, train_dist, test_dist, all_test_indices, all_train_indices, train_means, train_maxs
end


function store_dv_data(all_activity, test_set_num, d, t, session_id_idx, i, i_code, train_Wasserstein, train_reference_performance, test_reference_performance, 
    train_validity_rate, test_Wasserstein_list, W, reward_values; n_clamp=3, constant_intervals=nothing, αs=nothing, δ=nothing, NW1=nothing, NW2=nothing, NW3=nothing)

    global res

    # line below doesn't work for quantile codes specifically
    # train_validity_rate, train_CDFs_x, train_CDFs_y = validity_rate_and_CDFs(train_predicted_CDFs2, reward_values)

    res[d][t, session_id_idx, i, i_code, 1, 1] = train_Wasserstein
    res[d][t, session_id_idx, i, i_code, 1, 2] = train_Wasserstein / train_reference_performance
    res[d][t, session_id_idx, i, i_code, 1, 3] = train_validity_rate
    for s in 1:test_set_num
        res[d][t, session_id_idx, i, i_code, s+1, 1] = test_Wasserstein_list[s]
        res[d][t, session_id_idx, i, i_code, s+1, 2] = test_Wasserstein_list[s] / test_reference_performance

        test_activity = all_activity[s]
        if ~any(isnan.(test_activity))
            if i_code == 1
                predicted_test_quantiles = W * test_activity
                test_validity_rate, test_inverse_CDFs_x, test_inverse_CDFs_y = quantiles_validity_rate_and_inverse_CDFs(predicted_test_quantiles, constant_intervals)
            elseif i_code == 2
                test_predicted_CDFs = expectile_decoding_predicted_CDFs(αs, W, test_activity, reward_values, δ)
                test_predicted_CDFs2 = vcat( test_predicted_CDFs, transpose([1.0 for i in 1:size(test_predicted_CDFs)[2]]) )
                test_validity_rate, test_CDFs_x, test_CDFs_y = validity_rate_and_CDFs(test_predicted_CDFs2, reward_values)
            else
                if i_code == 3
                    test_predicted_CDFs2 = PPC_decode_CDFs2(W, test_activity)
                elseif i_code == 4
                    test_predicted_CDFs2 = DDC_decode_CDFs2(W, test_activity)
                elseif i_code == 5
                    test_predicted_CDFs2 = network_decode_CDFs2(W, NW1, NW2, NW3, test_activity)
                end
                test_validity_rate, test_CDFs_x, test_CDFs_y = validity_rate_and_CDFs(test_predicted_CDFs2, reward_values)
            end
            res[d][t, session_id_idx, i, i_code, s+1, 3] = test_validity_rate
        else
            res[d][t, session_id_idx, i, i_code, s+1, 3] = NaN
        end 
    end
    return res
end


function store_data(d, t, session_id_idx, i, i_code, train_Wasserstein, test_Wasserstein, train_reference_performance, test_reference_performance; reward_values=nothing,
    train_predicted_CDFs2=nothing, test_predicted_CDFs2=nothing, predicted_train_quantiles=nothing, predicted_test_quantiles=nothing, constant_intervals=nothing)
    
    global res

    if i_code == 1
        train_validity_rate, train_inverse_CDFs_x, train_inverse_CDFs_y = quantiles_validity_rate_and_inverse_CDFs(predicted_train_quantiles, constant_intervals)
        test_validity_rate, test_inverse_CDFs_x, test_inverse_CDFs_y = quantiles_validity_rate_and_inverse_CDFs(predicted_test_quantiles, constant_intervals)
    else
        train_validity_rate, train_CDFs_x, train_CDFs_y = validity_rate_and_CDFs(train_predicted_CDFs2, reward_values)
        test_validity_rate, test_CDFs_x, test_CDFs_y = validity_rate_and_CDFs(test_predicted_CDFs2, reward_values)
    end

    res[d][t, session_id_idx, i, i_code, 1, 1] = train_Wasserstein
    res[d][t, session_id_idx, i, i_code, 1, 2] = train_Wasserstein / train_reference_performance
    res[d][t, session_id_idx, i, i_code, 1, 3] = train_validity_rate
    res[d][t, session_id_idx, i, i_code, 2, 1] = test_Wasserstein
    res[d][t, session_id_idx, i, i_code, 2, 2] = test_Wasserstein / test_reference_performance
    res[d][t, session_id_idx, i, i_code, 2, 3] = test_validity_rate

    return res
end


function store_pseudo_data(neuron_selection_idx, set_idx, i_code, train_Wasserstein, test_set_num, test_Wasserstein_list, train_reference_performance, test_reference_performance)

    global res

    res[neuron_selection_idx, set_idx, i_code, 1, 1] = train_Wasserstein
    res[neuron_selection_idx, set_idx, i_code, 1, 2] = train_Wasserstein / train_reference_performance
    for t in 1:test_set_num
        res[neuron_selection_idx, set_idx, i_code, t+1, 1] = test_Wasserstein_list[t]
        res[neuron_selection_idx, set_idx, i_code, t+1, 2] = test_Wasserstein_list[t] / test_reference_performance
    end

    return res

end



function get_type_avgs(one_per_trial, cum_n_trials_per_type)
    # println(cum_n_trials_per_type)
    n_trial_types = length(cum_n_trials_per_type)
    # println(n_trial_types)
    prep_cum_n_trials_per_type = zeros(Int, n_trial_types + 1)
    prep_cum_n_trials_per_type[2:end] = cum_n_trials_per_type
    # println(prep_cum_n_trials_per_type)
    trial_type_avgs = zeros(n_trial_types)
    # println(trial_type_avgs)
    for i in 1:n_trial_types
        trial_type_avgs[i] = mean(one_per_trial[(prep_cum_n_trials_per_type[i]+1):prep_cum_n_trials_per_type[i+1]])
    end
    # println(trial_type_avgs)
    return trial_type_avgs
end
