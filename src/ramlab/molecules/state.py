from typing import Any, Dict
import numpy as np


class State:
    state: Dict[str, Any]

    def __init__(self, **state):
        self.state = state

    def __getattr__(self, name: str) -> Any:
        # if name == "state":
        #     return super().__getattribute__(name)
        if name in self.__dict__["state"].keys():
            return self.__dict__["state"][name]
        else:
            raise AttributeError(
                f"No attribute '{name}' in {str(self)}.\nDid you add all necessary quantum numbers?"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "state":
            super().__setattr__(name, value)
        else:
            self.state[name] = value
        # try:
        #     super().__setattr__(name, value)
        # except:
        #     self.state[name] = value

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

    def __repr__(self):
        return f"State({', '.join([f'{k}' for k in self.state.keys()])}) with length {len(self)}"

    def __add__(self, other):
        return State(
            **{k: np.concatenate([self.state[k], other.state[k]]) for k in self.state}
        )
