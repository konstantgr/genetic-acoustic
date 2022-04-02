import mph

from typing import Dict
from utils import copy_project, get_config, plot2d
from simple_evolutionary_algorithms import differential_evolution_circles_scipy, transform_to_binary_list
from individuals import CircleIndividual
import os

mph.option('classkit', True)


def read_config(cfg: Dict):
    return (cfg['source_directory'], cfg['tmp_directory'],
            cfg['target_directory'], cfg['images_directory'])


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    config = get_config(filename=os.path.join(dirname, 'config.yml'))
    src, tmp, dst, images_dst = read_config(config)
    copy_project(src, tmp)

    client = mph.start(cores=2)  # client.clear() can be used after the run
    model = client.load(tmp)  # model.clear() can be used after the modeling
    # client.java.setDefaultGeometryKernel('comsol')  # must be tested

    model.parameter('max_freq', '1000[Hz]')
    model.parameter('min_freq', '100[Hz]')
    model.parameter('step', '100[Hz]')

    # Genetic Algorithm
    n_circle_grid = 3
    best_x, best_res = differential_evolution_circles_scipy(model, n=n_circle_grid)
    x = transform_to_binary_list(best_x)

    # Best individual
    ind = CircleIndividual(x, model=model)
    ind.create_model()
    ind.solve_geometry()

    plot2d(model, 'acpr.p_s', images_dst + '\\' + 'best_image.png')

    model.save(dst)
    print('Project saved successfully')
