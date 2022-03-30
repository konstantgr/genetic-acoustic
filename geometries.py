import numpy as np


def circles_grid(n, xlim, ylim): 
    x = np.linspace(*xlim, n, endpoint=True)
    y = np.linspace(*ylim, n, endpoint=True)

    return x, y


def squares_grid(n, xlim, ylim): 
    x = np.linspace(*xlim, n, endpoint=True)
    y = np.linspace(*ylim, n, endpoint=True)

    return x, y


def add_circle(name, x_i, y_j, geometry, selections, r, alpha):
    node = geometry.create("Circle", name=name)
    node.property("r", str(r))
    node.property("pos", [str(x_i), str(y_j)])
    node.property("pos", [str(x_i), str(y_j)])

    node_sel = selections.create('Box', name=name)
    node_sel.property('entitydim', 2)

    modified_r = r * alpha
    node_sel.property('xmin', f'+{x_i}-{modified_r}')
    node_sel.property('xmax', f'+{x_i}+{modified_r}')
    node_sel.property('ymin', f'+{y_j}-{modified_r}')
    node_sel.property('ymax', f'+{y_j}+{modified_r}')
    node_sel.property('condition', 'inside')

    return node, node_sel


def add_square(name, x_i, y_j, geometry, selections, width, alpha):
    node = geometry.create("Square", name=name)
    node.property("size", str(width))
    node.property("pos", [str(x_i), str(y_j)])
    node.property("pos", [str(x_i), str(y_j)])

    node_sel = selections.create('Box', name=name)
    node_sel.property('entitydim', 2)

    modified_width = width / 2 * alpha
    node_sel.property('xmin', f'+{x_i}-{modified_width}+{width / 2}')
    node_sel.property('xmax', f'+{x_i}+{modified_width}+{width / 2}')
    node_sel.property('ymin', f'+{y_j}-{modified_width}+{width / 2}')
    node_sel.property('ymax', f'+{y_j}+{modified_width}+{width / 2}')
    node_sel.property('condition', 'inside')

    return node, node_sel
