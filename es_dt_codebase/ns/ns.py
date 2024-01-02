# Entry point for other modules (provides aliasing).

from . import head


def ns(
    models,
    test_environment,
    num_of_workers,
    size_of_population,
    num_of_iterations,
    main_seed,
    max_archive_size,
    num_of_nearest_neighbors_to_average,
    noise_deviation,
    weight_decay_factor,
    batch_size,
    update_vbn_stats_probability,
    path_for_checkpoints,
    logging_path
):
    return head.head_process(
        models,
        test_environment,
        num_of_workers,
        size_of_population,
        num_of_iterations,
        main_seed,
        max_archive_size,
        num_of_nearest_neighbors_to_average,
        noise_deviation,
        weight_decay_factor,
        batch_size,
        update_vbn_stats_probability,
        path_for_checkpoints,
        logging_path
    )
