import logging

import mph
import numpy as np
from tqdm import tqdm

from utils import copy_project, clean, get_config, get_indices, make_unique
from geometries import circles_grid, squares_grid, add_circle, add_square
mph.option('classkit', True)

import pandas as pd


def evaluate_global_ev(dataset: mph.Node, evaluation: mph.Node) -> pd.DataFrame:
    #  https://github.com/MPh-py/MPh/blob/2b967b77352f9ce7effcd50ad4774bf5eaf731ea/mph/model.py#L425
    evaluation.property('data', dataset)
    java = evaluation.java
    results = np.array(java.getReal())
    if java.isComplex():
        results = results.astype('complex')
        results += 1j * np.array(java.getImag())
    return pd.DataFrame(data=results.T, columns=make_unique(evaluation.property('descr')))


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
    config = get_config('config.yml')

    src = config['source_directory']
    tmp = config['tmp_directory']
    dst = config['target_directory']
    images_dst = config['images_directory']

    copy_project(src, tmp)

    client = mph.start(cores=1) # client.clear() can be used after the run
    model = client.load(tmp) # model.clear() can be used after the modeling

    air = model/'selections'/'air'
    plastic = model/'selections'/'plastic'
    selections = model/'selections'
    amodel = model/'geometries'/'Geometry 1'

    initial_params = {
        'xlim': (-0.05, 0.05),
        'ylim': (-0.05, 0.05),
        'n': 10, 
    }
    size = initial_params['n']**2

    x, y = circles_grid(**initial_params)
    tau = abs(x[1] - x[0])
    radius = tau / 2
    # x, y = squares_grid(**initial_params)
    # tau = abs(x[1] - x[0])
    # width = tau

    alpha = 1.1

    clean(amodel, 'circle')
    clean(selections, 'circle')
    clean(amodel, 'square')
    clean(selections, 'square')

    node_selections = []
    idx = 0
    indices = get_indices(size, p=0.8)
    for x_i in tqdm(x):
        for y_j in y:
            name = f"circle_xi_{x_i}, yj_{y_j}"
            # name = f"square_xi_{x_i}, yj_{y_j}"

            if idx in list(indices[0]):
                node, node_sel = add_circle(
                    name, 
                    x_i, y_j, 
                    amodel, selections, 
                    radius, alpha
                )        
                node_selections.append(node_sel)
                # node, node_sel = add_square(
                #     name, 
                #     x_i, y_j, 
                #     amodel, selections, 
                #     width, alpha
                # )        
                # node_selections.append(node_sel)

            else:
                node_selections.append(None)

            idx += 1

    plastic.property('input', list(np.array(node_selections)[indices]))
    model.build(amodel)
    
    model.mesh()
    model.solve()

    evaluation = model / 'evaluations' / 'Global Evaluation 1'
    dataset = (model / 'datasets').children()[0]
    evaluation.property('data', dataset)

    logging.info(f'Evaluation : {evaluation.name()}')
    logging.info(f'Dataset: {dataset.name()}')
    logging.info(f'Solution: {dataset.property("solution")}')

    evaluate_global_ev(dataset, evaluation).to_csv('result.csv')
    plot2d(model, 'acpr.p_s', images_dst + '\\' 'image.png')

    model.save(dst)
    logging.info('Project saved successfully')
