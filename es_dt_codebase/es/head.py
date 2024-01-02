# Head process, which will spawn workers and lead the training.

from . import funcs
from es_utilities import utils
from es_utilities import multiprocessing_utils as mpu
from .worker import worker_process

import random
import time
import os
import multiprocessing as mp
from ctypes import c_bool

from tqdm import trange

# Forbid multithreading for Numpy.
N_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS

import numpy as np
import torch

# Forbid multithreading for PyTorch.
torch.set_num_threads(1)

# Forbidding multithreading for Numpy and PyTorch here will sadly take effect only in workers, for here in this process, numpy and torch are already initialized.
# However, it's what it does in workers that we mainly care about, that's where most work is done.
# So here it serves as a safeguard, if it is not dealt with in initial calling script.


def head_process(
    model,
    test_environment,
    num_of_workers,
    size_of_population,
    num_of_iterations,
    main_seed,
    noise_deviation,
    weight_decay_factor,
    batch_size,
    update_vbn_stats_probability,
    path_for_checkpoints,
    logging_path
):
    # There are some problems when forking, so I'm "hardwiring" the process starting method to be "spawn", which I know works.
    mp.set_start_method('spawn')
    
    # Set the seed
    utils.set_seed(main_seed)
    
    # Initialization of shared variables
    ## Control events
    start_new_iteration = mp.Event()
    start_update = mp.Event()
    
    ## Control arrays
    run_finished = mp.Array(c_bool, num_of_workers, lock=False)
    running = mp.Array(c_bool, num_of_workers, lock=False)
    update_finished = mp.Array(c_bool, num_of_workers, lock=False)
    
    ## Task declaration queue (Tasks are given in a form of integers as indices of the tasks (which are used to point into shared memory), and Nones as termination tokens for individual workers.)
    task_queue = mp.Queue()
    
    ## Array of seeds of the individual tasks
    seed_array = mp.Array("l", size_of_population, lock=False)
    
    ## Results reporting arrays (1 for plus noise, 2 for minus noise) and rank_weight array for update.
    fitness1 = mp.Array("f", size_of_population, lock=False)
    fitness2 = mp.Array("f", size_of_population, lock=False)
    rank_weights = mp.Array("f", size_of_population, lock=False)
    
    ## VBN-stats-related shared memory
    num_of_entries_per_state = 1
    state_shape = test_environment.state_shape
    for dim_size in state_shape:
        num_of_entries_per_state *= dim_size
        
    sum_of_encountered_states = mp.Array("d", num_of_entries_per_state)
    sum_of_squares_of_encountered_states = mp.Array("d", num_of_entries_per_state)
    count_of_encountered_states = mp.Value("l")
    
    ## Shared values concerning runtimes of evaluations
    runtime_last_iteration1 = mp.Array("l", size_of_population, lock=False)
    runtime_last_iteration2 = mp.Array("l", size_of_population, lock=False)
    max_runtime = mp.Value("l", lock=False)
    max_runtime.value = test_environment.timestep_limit if test_environment.timestep_limit is not None else 2147483647
    
    ## Shared noise table
    shared_noise_table = utils.SharedNoiseTable(noise_deviation, main_seed)
    
    # Pipe for communicating errors from workers to the head
    (receiver_connection, sender_connection) = mp.Pipe(duplex=False)
    
    # Workers initialization
    workers = [mpu.ErrorCommunicativeProcess(target=worker_process, args=(
        worker_index,
        model.clone(), # Somehow influenced each other, if not cloned, when tested.
        test_environment, # Did not influence each other, unlike the model, when tested.
        num_of_iterations,
        main_seed,
        weight_decay_factor,
        noise_deviation,
        batch_size,
        update_vbn_stats_probability,
        shared_noise_table,
        start_new_iteration,
        start_update,
        run_finished,
        running,
        update_finished,
        task_queue,
        seed_array,
        fitness1,
        fitness2,
        rank_weights,
        sum_of_encountered_states,
        sum_of_squares_of_encountered_states,
        count_of_encountered_states,
        runtime_last_iteration1,
        runtime_last_iteration2,
        max_runtime
    ), daemon=True, sender_connection=sender_connection) for worker_index in range(num_of_workers)]
    for worker in workers:
        worker.start()
        
    # Creating non-existent directories on both logging and checkpoint paths and other path processing
    logging_path = os.path.normpath(logging_path)
    path_for_checkpoints = os.path.normpath(path_for_checkpoints)
    
    os.makedirs(os.path.dirname(logging_path), exist_ok=True)
    os.makedirs(os.path.dirname(path_for_checkpoints), exist_ok=True)
        
    # Local variables
    last_evaluation_result, last_evaluation_runtime, best_yet_iteration, best_return_yet, corresponding_runtime = float("nan"), float("nan"), None, -float("inf"), float("nan")
    evaluation_path = logging_path + ".evaluations"
    evaluation_csv_path = logging_path + ".evaluations.csv"
    fitness_path = logging_path + ".fitness.csv"
    runtime_path = logging_path + ".runtime.csv"
    time_path = logging_path + ".time.csv"
    
    # Clearing logfiles
    if os.path.exists(fitness_path):
        os.remove(fitness_path)
        
    if os.path.exists(runtime_path):
        os.remove(runtime_path)
        
    if os.path.exists(time_path):
        os.remove(time_path)
        
    if os.path.exists(evaluation_path):
        os.remove(evaluation_path)
    
    if os.path.exists(evaluation_csv_path):
        os.remove(evaluation_csv_path)
        
    with open(evaluation_csv_path, "a") as log:
            log.write(f"Evaluation result\tBest yet result\n")
    
    # Iterations / Run of the program
    progress_bar = trange(num_of_iterations)
    for iteration in progress_bar:
        # Report progress
        progress_bar.set_description(f"Running iteration {iteration+1} | Best yet evaluation result (mean runtime) being obtained after iteration {best_yet_iteration} " + \
            f"- {best_return_yet:.4f} ({corresponding_runtime})")
        
        iteration_start_time = time.time()
        
        start_update.clear()
        
        # Set or reset the shared variables and declare tasks
        utils.set_seed(main_seed + iteration if main_seed is not None else None)
        for i in range(size_of_population):
            ## Declare tasks
            task_queue.put(i)
            seed_array[i] = random.randint(0, 2147483647)
        
        for i in range(num_of_workers):
            run_finished[i] = False
            update_finished[i] = False
            
            ## Signal / Mark the end of the task queue for the workers
            task_queue.put(None)
            
        count_of_encountered_states.value = 0
        sum_of_encountered_states[:] = [0] * num_of_entries_per_state
        sum_of_squares_of_encountered_states[:] = [0] * num_of_entries_per_state
            
        start_new_iteration.set()
        
        # Evaluate the model resulting from the previous iteration and save it, if it is better than the best yet encountered
        last_evaluation_result, last_evaluation_runtime = funcs.evaluate_and_possibly_save(
            model,
            test_environment,
            best_return_yet,
            10,
            path_for_checkpoints
        )

        if last_evaluation_result >= best_return_yet:
            best_return_yet = last_evaluation_result
            corresponding_runtime = last_evaluation_runtime
            best_yet_iteration = iteration

            progress_bar.set_description(f"Running iteration {iteration+1} | Best yet evaluation result (mean runtime) being obtained after iteration {best_yet_iteration} " + \
                f"- {best_return_yet:.4f} ({corresponding_runtime})")

        # Log the evaluation results
        with open(evaluation_path, "a") as log:
            log.write(f"Iteration {iteration} - Evaluation result (mean runtime): {last_evaluation_result} ({last_evaluation_runtime}) | Best yet: {best_return_yet} ({corresponding_runtime})\n")

        with open(evaluation_csv_path, "a") as log:
            log.write(f"{last_evaluation_result}\t{best_return_yet}\n")
                
        while not all(run_finished):
            mpu.check_for_errors_and_wait(receiver_connection)
            
        # Logging - fitness + runtime
        ## Fitness
        funcs.log_iteration_population_data(fitness_path, fitness1, fitness2)
            
        ## Runtime
        funcs.log_iteration_population_data(runtime_path, runtime_last_iteration1, runtime_last_iteration2)
        
        # Change fitness to the score / weight for update.
        modified_fitnesses = np.array(fitness1[:]) - np.array(fitness2[:])
        modified_fitnesses[modified_fitnesses.argsort()] = np.arange(size_of_population) # From interval [0, size_of_population-1]
        modified_fitnesses /= (size_of_population - 1) # From interval [0,1]
        modified_fitnesses *= 2 # From interval [0,2]
        modified_fitnesses -= 1 # From interval [-1,1]
        rank_weights[:] = modified_fitnesses
        
        start_new_iteration.clear()
        start_update.set()
        
        funcs.update(
            None,
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
        )
        
        # Update max_runtime
        total_runtime_last_iteration = sum(runtime_last_iteration1) + sum(runtime_last_iteration2)
        max_runtime.value = total_runtime_last_iteration // size_of_population # = twice the mean number of steps taken per episode
        
        while not all(update_finished):
            mpu.check_for_errors_and_wait(receiver_connection)
            
        # Logging - iteration wall-clock duration
        iteration_duration = time.time() - iteration_start_time
        with open(time_path, "a") as log:
            log.write(str(iteration_duration) + "\n")
            
    # Final evaluation and saving of the resulting model
    last_evaluation_result, last_evaluation_runtime = funcs.evaluate_and_possibly_save(
        model,
        test_environment,
        best_return_yet,
        10,
        path_for_checkpoints
    )

    if last_evaluation_result >= best_return_yet:
        best_return_yet = last_evaluation_result
        corresponding_runtime = last_evaluation_runtime
    
    model.save_parameters(path_for_checkpoints, "final_model")
    
    print(f"Best evaluation result (mean runtime): {best_return_yet} ({corresponding_runtime})")
            
    return model
