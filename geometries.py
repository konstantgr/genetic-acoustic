import numpy as np

def circles_grid(n, xlim, ylim): 
    size = 10
    x = np.linspace(*xlim, n, endpoint=True) / 10
    y = np.linspace(*ylim, n, endpoint=True) / 10

    return x, y


def add_circle(name, x_i, y_j, amodel, selections, r, alpha):
    name = f"circle_xi_{x_i}, yj_{y_j}"
    node = amodel.create("Circle", name=name)
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
