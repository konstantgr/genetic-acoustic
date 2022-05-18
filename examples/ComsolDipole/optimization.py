from pckit import Task
import pandas as pd


def get_scattering_from_result(result: pd.DataFrame):
    return result['sigma'][0]


def fitness(result: pd.DataFrame):
    # просто возвращает сечение рассеяния
    return get_scattering_from_result(result)


def main(solver):
    tasks = [
        Task(cylinders_h=[600, 600],  # высоты, в данной модели будет 600 для всех
             cylinders_r=[200, 300],
             cylinders_seps=[100, 100],
             save=False),

        Task(cylinders_h=[600, 600],
             cylinders_r=[200, 200],
             cylinders_seps=[100, 100],
             save=True)
    ]
    results = solver.solve(tasks)
    print([fitness(res) for res in results])
