import numpy as np
import pandas as pd
from ramlab.molecules.ab_initio_molecule import AbInitioMolecule
from ramlab.molecules.state import State
from ramlab.molecules.transitions import Transitions


class NO(AbInitioMolecule):
    @classmethod
    def E(cls, state: State) -> float:
        # TODO: Implement the energy of the NO molecule.
        # You can use expressions like state.v and state.J to access the vibrational and rotational quantum numbers.
        raise NotImplementedError()

    @classmethod
    def _get_all_transition_states(cls) -> tuple[State, State]:
        # TODO: Implement the initial and final states for all possible transitions.
        # You can use the State class to create the states.
        # e.g.
        #   state_initial = State(v=0, J=0)
        #   state_final = State(v=1, J=1)
        #   return state_initial, state_final
        raise NotImplementedError()

    @classmethod
    def _format_quanta_global(cls, state: State):
        # TODO: Implement the formatting of the global quanta.
        # You can use expressions like np.char.mod("%2d", state.v) to format the vibrational quantum number.
        raise NotImplementedError()

    @classmethod
    def _format_quanta_local(cls, state: State):
        # TODO: Implement the formatting of the local quanta.
        # You can use expressions like np.char.mod("%2d", state.J) to format the rotational quantum number.
        raise NotImplementedError()

    @classmethod
    def process_hitran_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        # Parse the quantum numbers from the HITRAN format for local/global quantum numbers
        # e.g.
        #   df["lower_v"] = df["lower_quanta_global"].str[0:2].astype("Int64")
        raise NotImplementedError()

    @classmethod
    def _validate_transitions(cls, transitions: Transitions):
        return super()._validate_transitions(transitions)

    @classmethod
    def _calc_degeneracy(cls, state: State):
        return 1

    @classmethod
    def _calc_crosssection(cls, transitions: Transitions):
        return 1

    @classmethod
    def _calc_depolarization_ratio(cls, transitions: Transitions):
        return 1
