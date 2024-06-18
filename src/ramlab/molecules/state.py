from typing import Any, Dict
import numpy as np


class State:
    state: Dict[str, Any]

    def __init__(self, **state):
        self.state = state
        

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__.get("state", {}):
            return self.__dict__["state"][name]
        else:
            try:
                return super().__getattribute__(name)
            except AttributeError:
                raise AttributeError(
                    f"No attribute '{name}' in {str(self)}.\nDid you add all necessary quantum numbers?"
                )

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "state":
            super().__setattr__(name, value)
        else:
            self.state[name] = value
    

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
        repr_strs = []
        for k, v in self.__dict__.items():
            if isinstance(v, dict) and 'value' in v and 'units' in v:
                repr_strs.append(f'{k}: {v["value"]} {v["units"]}')
            else:
                repr_strs.append(f'{k}: {str(v)}')
        return f"State({', '.join(repr_strs)})"

    def __add__(self, other):
        return State(
            **{k: np.concatenate([self.state[k], other.state[k]]) for k in self.state}
        )
