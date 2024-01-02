# Functions for main loops of the processes.

from es_utilities import utils

# The following methods do not differ between es and ns, hence we use the original implementations.
from es.funcs import prepare_for_evaluations, finish_evaluations_part_of_iteration, update, evaluate_and_possibly_save, log_iteration_population_data

import random

import numpy as np


def prepare_iteration(worker_index, running_array, archive, last_updated_model_index, behaviors_of_models):
    if last_updated_model_index is not None:
        archive.add(behaviors_of_models[last_updated_model_index])
    running_array[worker_index] = True
    

def evaluations(
    task_index,
    model,
    test_model,
    test_environment,
    archive,
    seed_array,
    update_vbn_stats_probability,
    shared_noise_table,
    score1,
    score2,
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
    behavior_characteristic, runtime, sum, sum_of_squares, count = evaluation(test_model, test_environment, max_runtime, update_vbn_stats)
    
    # Update shared variables based on results of evaluation with +noise
    update_shared_variables(
        task_index,
        archive,
        score1,
        runtime_last_iteration1,
        sum_of_encountered_states,
        sum_of_squares_of_encountered_states,
        count_of_encountered_states,
        behavior_characteristic,
        runtime,
        sum,
        sum_of_squares,
        count
    )
    
    # Run with -noise
    test_model.copy_from(model)
    utils.subtract_noise(test_model, noise)
    
    update_vbn_stats = (random.random() < update_vbn_stats_probability)
    behavior_characteristic, runtime, sum, sum_of_squares, count = evaluation(test_model, test_environment, max_runtime, update_vbn_stats)
    
    # Update shared variables based on results of evaluation with -noise
    update_shared_variables(
        task_index,
        archive,
        score2,
        runtime_last_iteration2,
        sum_of_encountered_states,
        sum_of_squares_of_encountered_states,
        count_of_encountered_states,
        behavior_characteristic,
        runtime,
        sum,
        sum_of_squares,
        count
    )

    
def evaluation(
    test_model,
    test_environment,
    max_runtime,
    store_vbn_stats
):
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
    
    episode_return, episode_length = 0, 0
    
    if store_vbn_stats:
        observed_states = list()
    
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
    
    behavior_characteristic = test_environment.get_behavior_characteristic()
    
    if store_vbn_stats:
        observed_states = np.array(observed_states)
        sum_ = observed_states.sum(axis=0).reshape(-1)
        sum_of_squares = np.square(observed_states).sum(axis=0).reshape(-1)
        count = len(observed_states)
    else:
        sum_ = None
        sum_of_squares = None
        count = None
    
    return behavior_characteristic, episode_length, sum_, sum_of_squares, count


def update_shared_variables(
    task_index,
    archive,
    shared_score,
    shared_runtime,
    shared_sum,
    shared_sum_of_squares,
    shared_count,
    behavior_characteristic,
    runtime,
    sum,
    sum_of_squares,
    count
):
    novelty_score = archive.get_novelty_score_for_behavior(behavior_characteristic)
    shared_score[task_index] = novelty_score
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
