import mph

from typing import Dict
from comsol.utils import copy_project, get_config, plot2d
from simple_evolutionary_algorithms import differential_evolution_scipy, transform_to_binary_list
from comsol.individuals import CircleIndividual, SquareIndividual
import os

mph.option('classkit', True)
dirname = os.path.dirname(__file__)


def read_path_config(cfg: Dict):
    paths = (os.path.join(dirname, cfg['source_directory']),
             os.path.join(dirname, cfg['tmp_directory']),
             os.path.join(dirname, cfg['target_directory']),
             os.path.join(dirname, cfg['images_directory']))
    for path in paths:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
    return paths


if __name__ == "__main__":
    config = get_config(filename=os.path.join(dirname, 'config.yml'))
    src, tmp, dst, images_dst = read_path_config(config)
    copy_project(src, tmp)

    client = mph.start(cores=os.cpu_count())  # client.clear() can be used after the run
    model = client.load(tmp)  # model.clear() can be used after the modeling
    # client.java.setDefaultGeometryKernel('comsol')  # must be tested

    model.parameter('max_freq', '8000[Hz]')
    model.parameter('min_freq', '100[Hz]')
    model.parameter('step', '100[Hz]')

    # Genetic Algorithm
    n_grid = 5
    best_x, best_res = differential_evolution_scipy(model, SquareIndividual, n=n_grid)
    x = transform_to_binary_list(best_x)

    # Best individual
    ind = SquareIndividual(x, model=model)
    ind.create_model()
    ind.solve_geometry()

    plot2d(model, 'acpr.p_s', images_dst)

    model.save(dst)
    print(f'Project saved successfully, best result: {best_res}')
