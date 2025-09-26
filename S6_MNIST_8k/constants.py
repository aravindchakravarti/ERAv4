from dataclasses import dataclass

@dataclass
class dataset_constants:
    dataset_mean: tuple = (0.1307,)
    dataset_std: tuple = (0.3081,)