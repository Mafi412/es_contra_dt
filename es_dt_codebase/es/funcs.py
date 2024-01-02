# Functions for main loops of the processes.

from es_utilities import utils

import random

import numpy as np


def prepare_iteration(worker_index, running_array):
    running_array[worker_index] = True


def prepare_for_evaluations(test_environment, seed):
    utils.set_seed(seed)
    test_environment.set_seed(seed)
    

def evaluations(
    task_index,
    model,
    test_model,
    test_environment,
    seed_array,
    update_vbn_stats_probability,
    shared_noise_table,
    fitness1,
    fitness2,
    sum_of_encountered_states,
    sum_of_squares_of_encountered_states,
    count_of_encountered_states,
    runtime_last_iteration1,
    runtime_last_iteration2,
    max_runtime
):
    # Get noise
    noise = utils.get_noise(model, shared_noise_table, seed_array[task_index])
    
    # Run with +noise
    test_model.copy_from(model)
    utils.add_noise(test_model, noise)
    
    update_vbn_stats = (random.random() < update_vbn_stats_probability)
    fitness, runtime, sum, sum_of_squares, count = evaluation(1, test_model, test_environment, max_runtime, update_vbn_stats)
    
    # Update shared variables based on results of evaluation with +noise
    update_shared_variables(
        task_index,
        fitness1,
        runtime_last_iteration1,
        sum_of_encountered_states,
        sum_of_squares_of_encountered_states,
        count_of_encountered_states,
        fitness,
        runtime,
        sum,
        sum_of_squares,
        count
    )
    
    # Run with -noise
    test_model.copy_from(model)
    utils.subtract_noise(test_model, noise)
    
    update_vbn_stats = (random.random() < update_vbn_stats_probability)
    fitness, runtime, sum, sum_of_squares, count = evaluation(1, test_model, test_environment, max_runtime, update_vbn_stats)
    
    # Update shared variables based on results of evaluation with -noise
    update_shared_variables(
        task_index,
        fitness2,
        runtime_last_iteration2,
        sum_of_encountered_states,
        sum_of_squares_of_encountered_states,
        count_of_encountered_states,
        fitness,
        runtime,
        sum,
        sum_of_squares,
        count
    )

    
def evaluation(
    num_of_episodes,
    test_model,
    test_environment,
    max_runtime,
    store_vbn_stats
):
    episode_returns, episode_lengths = list(), list()
    if store_vbn_stats:
        observed_states = list()
    
    if test_environment.timestep_limit is not None:
        if max_runtime is not None:
            max_timestep = min(test_environment.timestep_limit, max_runtime.value)
        else:
            max_timestep = test_environment.timestep_limit
    else:
        if max_runtime is not None:
            max_timestep = max_runtime.value
        else:
            max_timestep = int(1e18) # It's just, who would need or even want more timesteps then this...?
    
    for episode in range(num_of_episodes):
        episode_return, episode_length = 0, 0
        test_model.reset_inner_state()
        state = test_environment.reset()
        
        for timestep in range(max_timestep):
            action = test_model.choose_action(state)

            next_state, reward, terminated, truncated = test_environment.step(action)
            done = terminated or truncated

            test_model.update_after_step(state, next_state, action, reward, terminated, truncated)
            
            if store_vbn_stats:
                observed_states.append(state)
            
            state = next_state

            episode_return += reward
            episode_length += 1

            if done:
                break
            
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
       
    mean_return = sum(episode_returns) / len(episode_returns)
    mean_length = sum(episode_lengths) // len(episode_lengths)
    
    if store_vbn_stats:
        observed_states = np.array([np.array(o) for o in observed_states])
        sum_ = observed_states.sum(axis=0).reshape(-1)
        sum_of_squares = np.square(observed_states).sum(axis=0).reshape(-1)
        count = len(observed_states)
    else:
        sum_ = None
        sum_of_squares = None
        count = None
    
    return mean_return, mean_length, sum_, sum_of_squares, count


def update_shared_variables(
    task_index,
    shared_fitness,
    shared_runtime,
    shared_sum,
    shared_sum_of_squares,
    shared_count,
    fitness,
    runtime,
    sum,
    sum_of_squares,
    count
):
    shared_fitness[task_index] = fitness
    shared_runtime[task_index] = runtime
        
    if sum is not None:
        with shared_sum.get_lock():
            for i in range(len(sum)):
                shared_sum[i] += sum[i]
                
        with shared_sum_of_squares.get_lock():
            for i in range(len(sum_of_squares)):
                shared_sum_of_squares[i] += sum_of_squares[i]
                
        with shared_count.get_lock():
            shared_count.value += count


def finish_evaluations_part_of_iteration(running, run_finished, worker_index):
    # Update process status
    running[worker_index] = False
    run_finished[worker_index] = True


def update(
    worker_index,
    model,
    weight_decay_factor,
    noise_deviation,
    batch_size,
    shared_noise_table,
    update_finished,
    seed_array,
    rank_weights,
    sum_of_encountered_states,
    sum_of_squares_of_encountered_states,
    count_of_encountered_states,
):
    # Get and weight noises
    noises = list()
    for task_index in range(len(seed_array)):
        current_noise = utils.get_noise(model, shared_noise_table, seed_array[task_index])
        rank_weight = rank_weights[task_index]
        noises.append(utils.get_weighted_noise(rank_weight, current_noise))
        if len(noises) >= batch_size:
            combined_noise = utils.get_combined_noises(noises)
            noises = [combined_noise]
    combined_noise = utils.get_combined_noises(noises)
    
    # Update the model by combined noises
    ## The combined noise is to be divided by the number of individuals evaluated and the noise deviation squared.
    ## That is because in the original paper they divide by the number of individuals evaluated and the noise deviation, but use noise drawn from distribution with sd=1 (and only scale it during evaluation).
    ## We, on the other hand, use noise drawn from distribution with other sd, which is basically noise drawn with sd=1 multiplied by our sd. Hence we have to divide this excess sd.
    combined_noise = utils.get_weighted_noise(1 / (2 * len(seed_array) * (noise_deviation ** 2)), combined_noise)
    model.optimizer.update(combined_noise)
        
    # Weight decay
    utils.decay_weights(weight_decay_factor, model)
        
    # Update the virtual batch normalization stats
    shape = model.vbn_stats.shape
    model.vbn_stats.increment(
        np.array(sum_of_encountered_states).reshape(shape),
        np.array(sum_of_squares_of_encountered_states).reshape(shape),
        count_of_encountered_states.value
    )
    
    # Update process status
    if worker_index is not None: # Worker process, else it was called by the head.
        update_finished[worker_index] = True


def evaluate_and_possibly_save(model, test_environment, best_return_yet, num_of_episodes, path_for_checkpoints):
    last_evaluation_result, last_evaluation_runtime, _, _, _ = evaluation(num_of_episodes, model, test_environment, None, False)
    
    if last_evaluation_result >= best_return_yet:
        model.save_parameters(path_for_checkpoints, "best")
    
    return last_evaluation_result, last_evaluation_runtime


def log_iteration_population_data(path, data_to_log1, data_to_log2):
    with open(path, "a") as log:
        log.write(str(data_to_log1[0]) + ";" + str(data_to_log2[0]))
        for i in range(1, len(data_to_log1)):
            log.write(";" + str(data_to_log1[i]) + ";" + str(data_to_log2[i]))
        log.write("\n")
