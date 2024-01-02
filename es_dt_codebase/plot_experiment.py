from data_analysis import plots, dataloading

from argparse import ArgumentParser


def main(args):
    if args.names_of_individual_experiments is not None:
        assert len(args.paths_to_data) == len(args.names_of_individual_experiments)
    
    if args.experiment_type == "es":
        all_evaluation_fitnesses, all_fitnesses, all_runtimes, all_iteration_times = [], [], [], []
        
        print("Loading data...")
        for path in args.paths_to_data:
            evaluation_fitnesses, fitnesses, runtimes, iteration_times = dataloading.load_es_data(path, args.max_iterations)
            all_evaluation_fitnesses.append(evaluation_fitnesses)
            all_fitnesses.append(fitnesses)
            all_runtimes.append(runtimes)
            all_iteration_times.append(iteration_times)
            
        print("Plotting evaluation fitnesses...")
        plots.plot_evaluation_fitness(*all_evaluation_fitnesses, num_of_iterations_to_plot=args.max_iterations, values_range=(0, args.max_fitness), plot_dimensions=args.plot_dimensions, line_names=args.names_of_individual_experiments)
        print("Plotting fitnesses...")
        plots.plot_fitness(*all_fitnesses, num_of_iterations_to_plot=args.max_iterations, values_range=(0, args.max_fitness), plot_dimensions=args.plot_dimensions, line_names=args.names_of_individual_experiments)
        print("Plotting runtimes...")
        plots.plot_runtime(*all_runtimes, num_of_iterations_to_plot=args.max_iterations, plot_dimensions=args.plot_dimensions, line_names=args.names_of_individual_experiments)
        print("Plotting iteration wall-clock times...")
        plots.plot_time(*all_iteration_times, num_of_iterations_to_plot=args.max_iterations, plot_dimensions=args.plot_dimensions, line_names=args.names_of_individual_experiments)
        
        
    elif args.experiment_type == "ns":
        all_evaluation_fitnesses, all_novelty_scores, all_runtimes, all_iteration_times = [], [], [], []
        
        print("Loading data...")
        for path in args.paths_to_data:
            evaluation_fitnesses, novelty_scores, runtimes, iteration_times = dataloading.load_ns_data(path, args.max_iterations)
            all_evaluation_fitnesses.append(evaluation_fitnesses)
            all_novelty_scores.append(novelty_scores)
            all_runtimes.append(runtimes)
            all_iteration_times.append(iteration_times)
            
        print("Plotting evaluation fitnesses...")
        plots.plot_evaluation_fitness(*all_evaluation_fitnesses, num_of_iterations_to_plot=args.max_iterations, values_range=(0, args.max_fitness), plot_dimensions=args.plot_dimensions, line_names=args.names_of_individual_experiments)
        print("Plotting novelty scores...")
        plots.plot_novelty(*all_novelty_scores, num_of_iterations_to_plot=args.max_iterations, values_range=(0, args.max_novelty), plot_dimensions=args.plot_dimensions, line_names=args.names_of_individual_experiments)
        print("Plotting runtimes...")
        plots.plot_runtime(*all_runtimes, num_of_iterations_to_plot=args.max_iterations, plot_dimensions=args.plot_dimensions, line_names=args.names_of_individual_experiments)
        print("Plotting iteration wall-clock times...")
        plots.plot_time(*all_iteration_times, num_of_iterations_to_plot=args.max_iterations, plot_dimensions=args.plot_dimensions, line_names=args.names_of_individual_experiments)
        
        
    elif args.experiment_type == "qd":
        all_evaluation_fitnesses, all_fitnesses, all_novelty_scores, all_runtimes, all_iteration_times = [], [], [], [], []
        
        print("Loading data...")
        for path in args.paths_to_data:
            evaluation_fitnesses, fitnesses, novelty_scores, runtimes, iteration_times = dataloading.load_qd_data(path, args.max_iterations)
            all_evaluation_fitnesses.append(evaluation_fitnesses)
            all_fitnesses.append(fitnesses)
            all_novelty_scores.append(novelty_scores)
            all_runtimes.append(runtimes)
            all_iteration_times.append(iteration_times)
            
        print("Plotting evaluation fitnesses...")
        plots.plot_evaluation_fitness(*all_evaluation_fitnesses, num_of_iterations_to_plot=args.max_iterations, values_range=(0, args.max_fitness), plot_dimensions=args.plot_dimensions, line_names=args.names_of_individual_experiments)
        print("Plotting fitnesses...")
        plots.plot_fitness(*all_fitnesses, num_of_iterations_to_plot=args.max_iterations, values_range=(0, args.max_fitness), plot_dimensions=args.plot_dimensions, line_names=args.names_of_individual_experiments)
        print("Plotting novelty scores...")
        plots.plot_novelty(*all_novelty_scores, num_of_iterations_to_plot=args.max_iterations, values_range=(0, args.max_novelty), plot_dimensions=args.plot_dimensions, line_names=args.names_of_individual_experiments)
        print("Plotting runtimes...")
        plots.plot_runtime(*all_runtimes, num_of_iterations_to_plot=args.max_iterations, plot_dimensions=args.plot_dimensions, line_names=args.names_of_individual_experiments)
        print("Plotting iteration wall-clock times...")
        plots.plot_time(*all_iteration_times, num_of_iterations_to_plot=args.max_iterations, plot_dimensions=args.plot_dimensions, line_names=args.names_of_individual_experiments)
        
    
    else:
        raise ValueError("Invalid experiment type " + args.experiment_type)


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("paths_to_data", type=str, nargs="+", help="Path to the logged data. Should be without extensions, one per experiment to be plotted.")
    parser.add_argument("-t", "--experiment_type", type=str, default="es", help="Type of experiment. Either \"es\", or \"ns\", or \"qd\".")
    parser.add_argument("-i", "--max_iterations", type=int, default=200, help="Maximal number of iterations to be plotted.")
    parser.add_argument("-f", "--max_fitness", type=float, default=10, help="Maximal fitness value on the graphs.")
    parser.add_argument("-n", "--max_novelty", type=float, default=None, help="Maximal novelty value on the graphs.")
    parser.add_argument("-pd", "--plot_dimensions", nargs=2, type=float, default=(3.8, 2.7), help="Plot dimensions (two values, x and y).")
    parser.add_argument("-names","--names_of_individual_experiments", type=str, nargs="+", default=None, help="Names of the individual experiments passed in paths_to_data arguments. When used, have to have the same number of entries as paths_to_data.")
    
    main(parser.parse_args())
