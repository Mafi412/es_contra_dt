# Worker - will be spawned by head and will perform the training.

from . import funcs
from novelty_utilities.archive import BehaviorArchive


def worker_process(
    worker_index,
    models,
    test_environment,
    behaviors_of_models,
    num_of_iterations,
    main_seed,
    weight_decay_factor,
    noise_deviation,
    batch_size,
    update_vbn_stats_probability,
    max_archive_size,
    num_of_nearest_neighbors_to_average,
    shared_noise_table,
    start_new_iteration,
    start_update,
    run_finished,
    running,
    update_finished,
    task_queue,
    current_model_index,
    seed_array,
    score1,
    score2,
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
    
    test_model = models[0].clone()
    
    # Behavior archive
    archive = BehaviorArchive(max_archive_size, num_of_nearest_neighbors_to_average)
    
    # Add behaviors of current models to the archive
    archive.extend(behaviors_of_models)
    
    last_updated_model_index = None
    
    for iteration in range(num_of_iterations):
        start_new_iteration.wait()
        
        funcs.prepare_iteration(worker_index, running, archive, last_updated_model_index, behaviors_of_models)
        
        while True:
            # Tasks are given in a form of integers as indices of the tasks (which are used to point into shared memory), and Nones as termination tokens for individual workers.)
            task_index = task_queue.get()
            
            if task_index is None:
                break
            
            funcs.prepare_for_evaluations(test_environment, main_seed + iteration if main_seed is not None else None)
        
            funcs.evaluations(
                task_index,
                models[current_model_index.value],
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
            )
            
        funcs.finish_evaluations_part_of_iteration(running, run_finished, worker_index)
        
        start_update.wait()
        
        last_updated_model_index = current_model_index.value
        
        funcs.update(
            worker_index,
            models[current_model_index.value],
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
