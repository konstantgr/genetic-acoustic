import numpy as np
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
from loguru import logger

from comsol.individuals import CircleIndividual, SquareIndividual
from comsol.fitness_functions import high_peaks, max_sc, peaks_contribution

invidividuals_level = logger.level("individuals", no=38)
bests_level = logger.level("best", no=38, color="<green>")
logger.add('logs/logs_{time}.log', level='INFO')

fmt = "{time} | {level} |\t{message}"
logger.add('logs/individuals_{time}.log', format=fmt, level='individuals')

P_CROSSOVER= 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 100

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

solved = {}
info = {'iteration':0, 'best': np.Inf}


def fitness_function(x, model, IndividualType):
    global info

    if str(x) in solved:
        info['iteration'] += 1
        return solved[str(x)],

    ind = IndividualType(x, model=model)
    ind.create_model()
    print('Running')
    ind.solve_geometry()

    # res = ind.fitness(func=peaks_contribution, R=0.18, c=343, multipole_n=1)
    res = ind.fitness(func=high_peaks, R=0.18, c=343)

    individual_string = "".join(np.array(x).astype(str))
    if res < info['best']:
        info['best'] = res
        message = f"iteration {info['iteration']} | individual {individual_string} | result {round(res, 4)}"
        logger.log("best", message)

    info['iteration'] += 1
    logger.info(f"[BEST {round(info['best'], 4)}]\titeration {info['iteration']}\tindividual {individual_string}\tresult {round(res, 4)}\tcalculation_time {ind.getLastComputationTime() / 1000}")
    message = f"iteration {info['iteration']} | individual {individual_string} | result {round(res, 4)}"
    logger.log("individuals", message)

    info['iteration'] += 1

    return res,


def deap_algorithm(model, IndividualType, n=2):
    ONE_MAX_LENGTH = n**2
    POPULATION_SIZE = n**2 * 2

    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("zeroOrOne", random.randint, 0, 1)
    toolbox.register("individualCreator",
                    tools.initRepeat,
                    creator.Individual,
                    toolbox.zeroOrOne,
                    ONE_MAX_LENGTH)

    toolbox.register("populationCreator",
                    tools.initRepeat, list,
                    toolbox.individualCreator)
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    toolbox.register("evaluate", lambda x: fitness_function(x, model=model, IndividualType=IndividualType))
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit,
                    indpb=1.0 / ONE_MAX_LENGTH)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    population, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=P_CROSSOVER,
        mutpb=P_MUTATION,
        ngen=MAX_GENERATIONS,
        stats=stats,
        verbose=True
    )

    return population, logbook
