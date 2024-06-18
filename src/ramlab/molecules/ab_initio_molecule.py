from pathlib import Path
from ramlab.hitran.lineformatter import format_transitions
from ramlab.molecules.hitran_linelist_molecule import LineListMolecule
import pandas as pd
import numpy as np

from ramlab.dirs import dir_data
from ramlab.molecules.state import State
from ramlab.molecules.transitions import Transitions
from ramlab.util.decorators import abstractproperty


class AbInitioMolecule(LineListMolecule):
    """A class for molecules with ab initio linelists. The line list are not shipped with the package, but are generated.

    Subclasses must implement the following methods:
    - _make_transitions(laser_wavelength: float, state_initial: State, state_final: State) -> Transitions
    - _get_all_transitions() -> tuple[State, State]
    """

    @classmethod
    def get_linelist_file(cls, laser_wavelength: float = None) -> str:
        """Returns the path to the line list file.

        Returns:
            str: The path to the line list file.
        """
        return (
            dir_data
            / "ab_initio"
            / cls.__name__
            / f"lambda_{laser_wavelength*1e9:.2f}nm.txt"
        )

    @classmethod
    def _make_linelist_file(
        cls,
        laser_wavelength: float,
        state_initial: State = None,
        state_final: State = None,
    ) -> Transitions:
        if state_initial is None or state_final is None:
            state_initial, state_final = cls._get_all_transition_states()

        transitions = cls._make_transitions(
            laser_wavelength, state_initial, state_final
        )
        cls._save_hitran_linelist(transitions, cls.get_linelist_file(laser_wavelength))

        return transitions

    @classmethod
    def _save_hitran_linelist(cls, transitions: Transitions, path):
        hitran_format = cls._transitions_to_hitran(transitions)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for line in hitran_format.values:
                f.write(line + "\n")

    @abstractproperty
    def molecule_number(subclass) -> int:
        return subclass.molecule_number

    @abstractproperty
    def molecule_name(subclass) -> str:
        return subclass.molecule_name

    @abstractproperty
    def isotope_number(cls) -> int:
        """The isotope number according to the HITRAN database."""
        return 1 #placeholder - probably correct anyway

    @classmethod
    def _make_transitions(
        cls, laser_wavelength: float, state_initial: State, state_final: State
    ) -> Transitions:
        #print (state_initial, state_final)
        for state in [state_initial, state_final]:
            state.degeneracy = cls._calc_degeneracy(state)
            state.quanta_global = cls._format_quanta_global(state)
            state.quanta_local = cls._format_quanta_local(state)
            state.quanta = np.char.add(state.quanta_global, state.quanta_local)
            #state.Er = cls.Er(state)
            #state.Ev = cls.Ev(state)
            state.E = cls.E(state)
             
        initial_is_lower = state_initial.E < state_final.E
        state_lower = state_initial[initial_is_lower] + state_final[~initial_is_lower]
        state_upper = state_final[initial_is_lower] + state_initial[~initial_is_lower]     
        transitions = Transitions(pd.DataFrame())
        for col in state_lower.state.keys():
            transitions["lower_" + col] = state_lower.state[col]
            transitions["upper_" + col] = state_upper.state[col]
        print("transitions complete")
        transitions.dv = state_upper.v - state_lower.v
        transitions.dJ = state_upper.J - state_lower.J
        transitions.dE = state_upper.E - state_lower.E
      #  transitions.dEr = state_upper.Er - state_lower.Er
     #   transitions.dEv = state_upper.Ev - state_lower.Ev
        if state_upper.O is not None:
            
            transitions.dO = state_upper.O - state_lower.O

        # Filter out transitions that are not allowed
        transitions = cls._validate_transitions(transitions)
        state_lower, state_upper = transitions.state_initial, transitions.state_final

        transitions.vacuum_wavenumber = state_upper.E - state_lower.E
        transitions.crosssection = cls._calc_crosssection(transitions)
      #  transitions.depolarization_ratio = cls._calc_depolarization_ratio(transitions)
        transitions.molecule_number = cls.molecule_number
        transitions.isotope_number = cls.isotope_number
        print(transitions)
        return transitions

    @classmethod
    def _format_quanta_global(cls, state: State):
        """Format the global quanta in a (max) 15-character string.

        Args:
            state (State): the state to format
        """
        raise NotImplementedError()

    @classmethod
    def _format_quanta_local(cls, state: State):
        """Format the local quanta in a (max) 15-character string.

        Args:
            state (State): the state to format
        """
        raise NotImplementedError()

    @classmethod
    def _get_all_transition_states(cls) -> tuple[State, State]:
        raise NotImplementedError()

    @classmethod
    def _calc_degeneracy(cls, state: State):
        raise NotImplementedError()

    @classmethod
    def _calc_crosssection(cls, transitions: Transitions):
        raise NotImplementedError()

    @classmethod
    def _calc_depolarization_ratio(cls, transitions: Transitions):
        raise NotImplementedError()

    @classmethod
    def _validate_transitions(cls, transitions: Transitions):
        idx = (
            (transitions.state_final.J >= 0)
            & (transitions.state_final.v >= 0)
            & (transitions.state_initial.J >= 0)
            & (transitions.state_initial.v >= 0)
            & (transitions.state_initial.quanta != transitions.state_final.quanta)
        )
        return transitions[idx]

    @classmethod
    def _transitions_to_hitran(cls, transitions: Transitions) -> pd.Series:
        return format_transitions(transitions)
