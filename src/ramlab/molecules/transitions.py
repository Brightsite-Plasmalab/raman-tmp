from typing import Any, Dict
import numpy as np
import pandas as pd

from ramlab.molecules.state import State


class Transitions:
    linelist: pd.DataFrame

    def __init__(self, linelist):
        self.linelist = linelist

    @property
    def state_initial(self):
        return State(
            **{
                k.replace("lower_", ""): self.linelist[k].values
                for k in self.linelist.columns
                if "lower_" in k
            }
        )

    @property
    def state_final(self):
        return State(
            **{
                k.replace("upper_", ""): self.linelist[k].values
                for k in self.linelist.columns
                if "upper_" in k
            }
        )

    def sortby(self, column: str, **kwargs) -> "Transitions":
        return Transitions(self.linelist.sort_values(by=column, **kwargs))

    def __getattr__(self, name: str) -> Any:
        if name in self.linelist.columns:
            return self.linelist[name].values
        else:
            raise AttributeError(f"'Transitions' object has no attribute '{name}'.")

    # def __setattr__(self, name: str, value: Any):
    #     if name == "linelist":
    #         super().__setattr__(name, value)
    #     self.linelist[name] = value

    def __getitem__(self, key):
        if (
            isinstance(key, slice)
            or isinstance(key, int)
            or isinstance(key, np.ndarray)
        ):
            # Get the start, stop, and step from the slice
            return Transitions(self.linelist.iloc[key])
        else:
            raise TypeError(
                f"Invalid argument type `{type(key)}`. Use a slice or an integer."
            )

    def __len__(self):
        return self.linelist.size
