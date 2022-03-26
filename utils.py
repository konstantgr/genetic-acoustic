import shutil
import yaml


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
