# Entry point for other modules (provides aliasing).

from . import head


def es(
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
    return head.head_process(
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
    )
