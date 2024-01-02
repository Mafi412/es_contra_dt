# Worker - will be spawned by head and will perform the training.

from . import funcs


def worker_process(
    worker_index,
    model,
    test_environment,
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
):
    # Those are not needed, nor wanted from the workers (and e.g. gym does sometimes output them)
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    test_model = model.clone()
    
    for iteration in range(num_of_iterations):
        start_new_iteration.wait()
        
        funcs.prepare_iteration(worker_index, running)
        
        while True:
            # Tasks are given in a form of integers as indices of the tasks (which are used to point into shared memory), and Nones as termination tokens for individual workers.)
            task_index = task_queue.get()
            
            if task_index is None:
                break
            
            funcs.prepare_for_evaluations(test_environment, main_seed + iteration if main_seed is not None else None)
        
            funcs.evaluations(
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
            )
            
        funcs.finish_evaluations_part_of_iteration(running, run_finished, worker_index)
        
        start_update.wait()
        
        funcs.update(
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
        )
