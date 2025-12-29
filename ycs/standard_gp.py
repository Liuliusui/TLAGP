import copy
import multiprocessing
import os
from time import strftime, localtime, time

import numpy
from deap import gp, base, creator, tools
import operator
from math import exp
import random
from instance_generator_loader import read_instances_from_folder_old
from functools import partial
from my_gp import primitive_set, eval_fitness_single, cxOnePointLeafBiased
try:
    from ycs.gp_simulation import simulation_run, dyn_lookahead
except ImportError:
    from .gp_simulation import simulation_run, dyn_lookahead

import pickle

# Init toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=primitive_set, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=primitive_set)

toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", cxOnePointLeafBiased, termpb=0.1)
# toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=8)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=primitive_set)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter('height'), max_value=8))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter('height'), max_value=8))

# Init mstats
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
stats_height = tools.Statistics(lambda ind: ind.height)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, height=stats_height)
mstats.register("avg", lambda x: round(numpy.mean(x), 2))
mstats.register("std", lambda x: round(numpy.std(x), 2))
mstats.register("min", lambda x: round(numpy.min(x), 2))
mstats.register("max", lambda x: round(numpy.max(x), 2))

# Init logbook
logbook = tools.Logbook()
logbook.header = ['gen', 'time', 'nevals', 'test_fitness', 'fitness', 'size', 'height']
logbook.chapters['fitness'].header = ['min', 'max', 'avg', 'std']
logbook.chapters['size'].header = ['min', 'max', 'avg', 'std']
logbook.chapters['height'].header = ['min', 'max', 'avg', 'std']


def log(evo_file: str, ind_file: str, eval_str, individuals, gen: int):
    print(eval_str)
    f = open(evo_file, 'a+')
    f.write(eval_str + '\n')
    f.close()

    if individuals:
        f = open(ind_file, 'a+')
        f.write(str(gen) + '\n')
        for i in individuals:
            f.write(str(i) + '\n')
        f.write('\n')
        f.close()


def create_files():
    path = './standard_gp_log/'
    if not os.path.exists(path):
        os.makedirs(path)

    time = strftime("%Y_%m_%d %H_%M_%S", localtime())

    evo_file = time
    evo_file += '_evo'
    evo_file = path + evo_file + '.log'

    ind_file = time
    ind_file += '_ind'
    ind_file = path + ind_file + '.log'

    pickle_file = time
    pickle_file += '_pickle'
    pickle_file = path + pickle_file + '.pkl'

    return evo_file, ind_file, pickle_file


def genetic_operator(population, crossover_probability, mutation_probability):
    offspring = [toolbox.clone(ind) for ind in population]
    # Apply crossover, mutation and reproduction on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < crossover_probability:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            # remove the fitness value of the individual
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if offspring[i].fitness.valid:
            if random.uniform(0, 1 - crossover_probability) < mutation_probability:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

    return offspring


def eval_single_instance(instance, individual):
    obj, _, _ = simulation_run(instance['task_dict'], instance['time_adjacency_matrix'],
                               dyn_lookahead, [gp.compile(individual, primitive_set)])
    return obj


def calculate_best_individual_test_fitness(hof, testing_set):
    best_individual = hof[0]
    objs = toolbox.map(partial(eval_single_instance, individual=best_individual), testing_set)
    gaps = []
    for i, obj in enumerate(objs):
        best_obj = testing_set[i]['offline_optimal']
        gap = (obj - best_obj) / best_obj * 100
        gaps.append(gap)
    return numpy.mean(gaps)


def evolution(training_set, testing_set, seed):
    random.seed(seed)
    evo_file, ind_file, pickle_file = create_files()

    hof = tools.HallOfFame(5)

    # Register the evaluation function
    toolbox.register("evaluate", eval_fitness_single, data_set=training_set)

    # Record start time
    start_time = time()

    pool = multiprocessing.Pool(processes=4)
    toolbox.register('map', pool.map)

    # Store whole population in each generation
    population_in_generations = []

    population_size = 100

    pop = toolbox.population(n=population_size)

    crossover_probability, mutation_probability, number_of_generations = 0.8,0.2,90

    temp_start = time()
    # Evaluate the entire population
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print('Time to evaluate the entire population:', time() - temp_start)

    hof.update(pop)
    # Sort the pop based on their fitness, and remove the worst individuals
    pop.sort(key=lambda x: x.fitness.values[0])
    pop = pop[:population_size]

    population_in_generations.append(copy.deepcopy(pop))

    temp_start = time()
    # Calculate the best individual test fitness
    best_individual_test_fitness = calculate_best_individual_test_fitness(hof, testing_set)
    print('Time to calculate best individual test fitness:', time() - temp_start)

    record = mstats.compile(pop)
    logbook.record(gen=0, time=time() - start_time, nevals=len(pop),
                   test_fitness=best_individual_test_fitness, **record)
    log(evo_file, ind_file, logbook.stream, hof, 0)

    for gen in range(1, number_of_generations):
        # Selection
        offspring = toolbox.select(pop, len(pop))

        # Genetic operators
        offspring = genetic_operator(offspring, crossover_probability, mutation_probability)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        hof.update(offspring)

        # Elitism
        offspring.extend(hof.items)
        # Sort the offspring based on their fitness, and remove the worst individuals (elitism number of individuals)
        offspring.sort(key=lambda x: x.fitness.values[0])
        offspring = offspring[:population_size]

        # Replace the current population by the offspring
        pop[:] = offspring

        population_in_generations.append(copy.deepcopy(pop))

        # Calculate the best individual test fitness
        best_individual_test_fitness = calculate_best_individual_test_fitness(hof, testing_set)

        # Append the current generation statistics to the logbook
        record = mstats.compile(pop) if mstats else {}
        logbook.record(gen=gen, time=time() - start_time, nevals=len(invalid_ind),
                       test_fitness=best_individual_test_fitness, **record)
        log(evo_file, ind_file, logbook.stream, hof, gen)

    with open(pickle_file, 'wb') as f:
        pickle.dump(population_in_generations, f)
    total_evo_time = time() - start_time
    print(f"[TIME] evolution_wall = {total_evo_time:.2f}s")
    return


def main():
    seed =13
    train_instances_old = read_instances_from_folder_old('ycs/train_instances')
    test_instances_old = read_instances_from_folder_old('ycs/test_instances')


    evolution(train_instances_old, test_instances_old, seed)


if __name__ == "__main__":
    main()
