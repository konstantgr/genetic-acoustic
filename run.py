import mph
import numpy as np
from tqdm import tqdm

from utils import copy_project, clean, get_config
from geometries import circles_grid, squares_grid, add_circle, add_square
mph.option('classkit', True)


def get_indices(size, p=0.5):
    np.random.seed(42)

    is_plastic = np.random.choice([0, 1], size=(size,), p=[1-p, p])
    return np.nonzero(is_plastic)


if __name__ == "__main__":
    config = get_config('config.yml')

    src = config['source_directory']
    tmp = config['tmp_directory']
    dst = config['target_directory']
    images_dst = config['images_directory']

    copy_project(src, tmp)

    client = mph.start(cores=1)
    model = client.load(tmp)

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

    # plots = model/'plots'
    # plots.java.setOnlyPlotWhenRequested(True)
    # plot = plots.create('PlotGroup2D', name='geom')

    # surface = plot.create('Surface', name='field strength')
    # surface.property('resolution', 'normal')
    # surface.property('expr', 'acpr.p_s')

    # exports = model/'exports'
    
    # image = exports.create('Image', name='image')
    # image.property('sourceobject', plots/'geom')
    # image.property('filename', images_dst + '\\' 'image.png')
    # image.property('size', 'manualweb')
    # image.property('unit', 'px')
    # image.property('height', '720')
    # image.property('width', '720')
    # model.export()

    model.save(dst)
    print('Project saved succesfully')
