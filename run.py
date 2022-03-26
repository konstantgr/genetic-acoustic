import mph
import shutil
import numpy as np

from utils import copy_project, clean
from geometries import circles_grid, add_circle
mph.option('classkit', True)


def get_indices(size, p=0.5):
    is_plastic = np.random.choice([0, 1], size=(size,), p=[1-p, p])
    return np.nonzero(is_plastic)


if __name__ == "__main__":
    src = r'F:\konstantin.grotov\konstantin.grotov\genetic-acoustics-materials\empty_project\empty_project.mph'
    tmp = r'F:\konstantin.grotov\konstantin.grotov\genetic-acoustics-materials\samples\tmp.mph'
    dst = r'F:\konstantin.grotov\konstantin.grotov\genetic-acoustics-materials\samples\sample1.mph'

    copy_project(src, tmp)

    client = mph.start(cores=1)
    model = client.load(tmp)

    air = model/'selections'/'air'
    plastic = model/'selections'/'plastic'
    selections = model/'selections'
    amodel = model/'geometries'/'Geometry 1'

    initial_params = {
        xlim: (-0.1, 0.1),
        ylim: (-0.1, 0.1),
        n: 20, 
    }
    size = initial_params['n']**2

    x, y = circles_grid(**initial_params)
    tau = abs(x[1] - x[0])
    radius = tau / 4
    alpha = 1.1

    clean(amodel, 'circle')
    clean(selections, 'circle')

    node_selections = []
    for x_i in x:
        for y_j in y:
            name = f"circle_xi_{x_i}, yj_{y_j}"

            node, node_sel = add_circle(name, x_i, y_j, amodel, selections, r, alpha)        
            node_selections.append(node_sel)

    plastic.property('input', list(np.array(node_selections)[get_indices(size)]))
    model.build(amodel)

    model.save(dst)
    print('Project saved succesfully')
