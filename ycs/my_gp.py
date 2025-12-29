import numpy
from deap import gp, base, creator, tools
import operator
from math import exp
import random
from deap.gp import compile

try:
    from ycs.gp_simulation import simulation_run, dyn_lookahead
except ImportError:
    from .gp_simulation import simulation_run, dyn_lookahead
import numpy as np

from collections import defaultdict
from functools import partial
from operator import eq, lt
import random

def protected_div(left, right):
    if abs(right) < 1e-7 or not np.isfinite(right):
        return 1.0
    else:
        return left / right


def protected_exp(x):
    if x <= 700:
        return exp(x)
    else:
        return float('inf')


def max_0(x):
    return max(x, 0)


# Terminal set
terminal_set = [
    'r_i', 'p_i', 's_ji', 'd_i', 'w_i', 't', 'n', 'bar_r', 'bar_p', 'bar_s', 'bar_d', 'bar_w'
]

primitive_set = gp.PrimitiveSet("MAIN", len(terminal_set), prefix='x')

# Rename the terminals
rename_dict = {}
for i in range(len(terminal_set)):
    rename_dict['x' + str(i)] = terminal_set[i]
primitive_set.renameArguments(**rename_dict)

# Define the function set
primitive_set.addPrimitive(operator.add, 2)
primitive_set.addPrimitive(operator.sub, 2)
primitive_set.addPrimitive(operator.mul, 2)
primitive_set.addPrimitive(protected_div, 2, name='div')
primitive_set.addPrimitive(max, 2)
primitive_set.addPrimitive(operator.neg, 1)
primitive_set.addPrimitive(max_0, 1)
primitive_set.addPrimitive(protected_exp, 1, name='exp')

# Constants terminal
# primitive_set.addEphemeralConstant("rand101",
# lambda: random.choice([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]))
# primitive_set.addEphemeralConstant("rand10", partial(lambda: round(random.uniform(0, 10), 3)))


def round_3_constant():
    return round(random.uniform(0, 10), 3)


primitive_set.addEphemeralConstant("rand10", round_3_constant)

# Init creator
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=primitive_set)


def init_mstat():
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", lambda x: round(numpy.mean(x), 2))
    mstats.register("std", lambda x: round(numpy.std(x), 2))
    mstats.register("min", lambda x: round(numpy.min(x), 2))
    mstats.register("max", lambda x: round(numpy.max(x), 2))
    return mstats


# Define the fitness function
def eval_fitness_single(individual, data_set):
    # Transform the tree expression to an executable function
    rule = compile(expr=individual, pset=primitive_set)

    # Calculate the fitness value
    fitness = []
    for instance in data_set:
        twt, task_sequence, _ = simulation_run(instance['task_dict'], instance['time_adjacency_matrix'],
                                            dyn_lookahead, [rule])
        fitness.append((twt - instance['offline_optimal']) / instance['offline_optimal'])
    return numpy.mean(fitness) * 100,


def eval_fitness_single_compiled(rule, data_set):
    # Calculate the fitness value
    fitness = []
    for instance in data_set:
        twt, task_sequence, _ = simulation_run(instance['task_dict'], instance['time_adjacency_matrix'],
                                            dyn_lookahead, [rule])
        fitness.append((twt - instance['offline_optimal']) / instance['offline_optimal'])
    return numpy.mean(fitness) * 100,


def eval_fitness_ensemble_compiled(rules, data_set):
    # Calculate the fitness value
    fitness = []
    for instance in data_set:
        twt, task_sequence, _ = simulation_run(instance['task_dict'], instance['time_adjacency_matrix'],
                                            dyn_lookahead, rules)
        fitness.append((twt - instance['offline_optimal']) / instance['offline_optimal'])
    return numpy.mean(fitness) * 100,


def cxOnePointLeafBiased(ind1, ind2, termpb):
    """Randomly select crossover point in each individual and exchange each
    subtree with the point as root between each individual.

    :param ind1: First typed tree participating in the crossover.
    :param ind2: Second typed tree participating in the crossover.
    :param termpb: The probability of choosing a terminal node (leaf).
    :returns: A tuple of two typed trees.

    When the nodes are strongly typed, the operator makes sure the
    second node type corresponds to the first node type.

    The parameter *termpb* sets the probability to choose between a terminal
    or non-terminal crossover point. For instance, as defined by Koza, non-
    terminal primitives are selected for 90% of the crossover points, and
    terminals for 10%, so *termpb* should be set to 0.1.
    """

    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2

    # Determine whether to keep terminals or primitives for each individual
    terminal_op = partial(eq, 0)
    primitive_op = partial(lt, 0)
    arity_op1 = terminal_op if random.random() < termpb else primitive_op
    arity_op2 = terminal_op if random.random() < termpb else primitive_op

    # List all available primitive or terminal types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)

    for idx, node in enumerate(ind1[1:], 1):
        if arity_op1(node.arity):
            types1[node.ret].append(idx)

    for idx, node in enumerate(ind2[1:], 1):
        if arity_op2(node.arity):
            types2[node.ret].append(idx)

    common_types = set(types1.keys()).intersection(set(types2.keys()))

    if len(common_types) > 0:
        # Set does not support indexing
        type_ = random.choice(list(common_types))
        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2
