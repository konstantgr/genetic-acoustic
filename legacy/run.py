import logging
import mph
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import copy_project, clean, get_config, get_indices, make_unique, evaluate_global_ev
from geometries import circles_grid, squares_grid, add_circle, add_square
from genetic_sample import simple_genetic, transform_to_binary, Individual

mph.option('classkit', True)


def plot2d(model: mph.Model, expr: str, filepath, props: dict = None):
    plots = model / 'plots'
    plots.java.setOnlyPlotWhenRequested(True)
    plot = plots.create('PlotGroup2D')

    surface = plot.create('Surface', name='field strength')
    surface.property('resolution', 'normal')
    surface.property('expr', expr)

    exports = model / 'exports'

    image = exports.create('Image')
    image.property('sourceobject', plot)
    image.property('filename', filepath)

    default_props = {'size': 'manualweb',
                     'unit': 'px',
                     'height': '720',
                     'width': '720'}
    for prop in default_props:
        image.property(prop, default_props[prop])
    if props is not None:
        for prop in props:
            image.property(prop, props[prop])
    model.export()
    image.remove()
    plot.remove()


if __name__ == "__main__":
    config = get_config('../config.yml')

    src = config['source_directory']
    tmp = config['tmp_directory']
    dst = config['target_directory']
    images_dst = config['images_directory']

    copy_project(src, tmp)

    client = mph.start(cores=8) # client.clear() can be used after the run
    model = client.load(tmp) # model.clear() can be used after the modeling
    # client.java.setDefaultGeometryKernel('comsol')  # must be tested

    model.parameter('max_freq', '10000[Hz]')
    model.parameter('min_freq', '100[Hz]')
    model.parameter('step', '100[Hz]')

    best_x, best_res = simple_genetic(model, n=3)
    x = transform_to_binary(best_x)
    ind = Individual(x, model=model)
    ind.create_model()
    ind.solve()
    # air = model/'selections'/'air'
    # plastic = model/'selections'/'plastic'
    # selections = model/'selections'
    # amodel = model/'geometries'/'Geometry 1'

    # initial_params = {
    #     'xlim': (-0.05, 0.05),
    #     'ylim': (-0.05, 0.05),
    #     'n': 10, 
    # }
    # size = initial_params['n']**2

    # x, y = circles_grid(**initial_params)
    # tau = abs(x[1] - x[0])
    # radius = tau / 2
    # # x, y = squares_grid(**initial_params)
    # # tau = abs(x[1] - x[0])
    # # width = tau

    # alpha = 1.1

    # clean(amodel, 'circle')
    # clean(selections, 'circle')
    # clean(amodel, 'square')
    # clean(selections, 'square')

    # node_selections = []
    # idx = 0
    # indices = get_indices(size, p=0.8)
    # for x_i in tqdm(x):
    #     for y_j in y:
    #         name = f"circle_xi_{x_i}, yj_{y_j}"
    #         # name = f"square_xi_{x_i}, yj_{y_j}"

    #         if idx in list(indices[0]):
    #             node, node_sel = add_circle(
    #                 name, 
    #                 x_i, y_j, 
    #                 amodel, selections, 
    #                 radius, alpha
    #             )        
    #             node_selections.append(node_sel)
    #             # node, node_sel = add_square(
    #             #     name, 
    #             #     x_i, y_j, 
    #             #     amodel, selections, 
    #             #     width, alpha
    #             # )        
    #             # node_selections.append(node_sel)

    #         else:
    #             node_selections.append(None)

    #         idx += 1

    # plastic.property('input', list(np.array(node_selections)[indices]))
    # model.build(amodel)
    
    # model.mesh()
    # model.solve()

    # evaluation = model / 'evaluations' / 'Global Evaluation 1'
    # dataset = (model / 'datasets').children()[0]
    # evaluation.property('data', dataset)

    # print(f'Evaluation : {evaluation.name()}')
    # print(f'Dataset: {dataset.name()}')
    # print(f'Solution: {dataset.property("solution")}')

    # evaluate_global_ev(dataset, evaluation).to_csv('result.csv')
    plot2d(model, 'acpr.p_s', images_dst + '\\' 'best_image.png')

    model.save(dst)
    print('Project saved successfully')
