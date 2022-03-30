import shutil
import yaml
import numpy as np
import mph
import pandas as pd


def copy_project(src, dst):
    print(f"copying\n{src}\n->\n{dst}")
    shutil.copyfile(src, dst)
    print('Success')


def clean(obj, flag: str):
    cnt = 0
    for c in obj.children():
        if flag in c.path[-1]:
            c.remove()
            cnt += 1

    print(f'Removed {cnt} objects')


def get_indices(size, p=0.5):
    np.random.seed(42)

    is_plastic = np.random.choice([0, 1], size=(size,), p=[1-p, p])
    return np.nonzero(is_plastic)
    

def get_config(filename):
    with open(filename) as f:
        return yaml.safe_load(f)


def make_unique(labels) -> list:
    new_labels = []
    _, real_index, counts = np.unique(labels, return_counts=True, return_inverse=True)
    for index in range(len(labels)):
        for count in range(counts[real_index[index]]):
            new_label = labels[index] + (f'({count})' if count != 0 else '')
            if new_label not in new_labels:
                new_labels.append(new_label)
                break
    return new_labels


def evaluate_global_ev(dataset: mph.Node, evaluation: mph.Node) -> pd.DataFrame:
    #  https://github.com/MPh-py/MPh/blob/2b967b77352f9ce7effcd50ad4774bf5eaf731ea/mph/model.py#L425
    evaluation.property('data', dataset)
    java = evaluation.java
    real, imag = java.computeResult()
    results = np.array(real) + 1j * np.array(imag)
    return pd.DataFrame(data=results, columns=make_unique(evaluation.property('descr')))
