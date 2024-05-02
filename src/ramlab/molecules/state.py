from typing import Any, Dict
import numpy as np


class State:
    state: Dict[str, Any]

    def __init__(self, **state):
        self.state = state

    def __getattr__(cls, name: str) -> Any:
        if name in cls.state:
            return cls.state[name]
        else:
            raise AttributeError(
                f"'{cls.__name__}' object has no attribute '{name}'. Did you add all necessary quantum numbers?"
            )

    def __getitem__(self, key):
        if (
            isinstance(key, slice)
            or isinstance(key, int)
            or isinstance(key, np.ndarray)
        ):
            # Get the start, stop, and step from the slice
            return State(**{k: self.state[k][key] for k in list(self.state.keys())})
        else:
            raise TypeError(
                f"Invalid argument type `{type(key)}`. Use a slice or an integer."
            )

    def __len__(self):
        return np.size(list(self.state.values())[0])
