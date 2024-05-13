import pandas as pd
from ramlab.hitran.parser import parse_hitran_data
from ramlab.molecules.base import Molecule
from ramlab.molecules.state import State
from ramlab.molecules.transitions import Transitions
from ramlab.dirs import dir_data


class LineListMolecule(Molecule):

    @classmethod
    def _has_linelist_file(cls, laser_wavelength: float) -> bool:
        return cls.get_linelist_file(laser_wavelength).exists()

    @classmethod
    def get_linelist_file(cls, laser_wavelength: float = None) -> str:
        """Returns the path to the line list file.

        Returns:
            str: The path to the line list file.
        """
        return dir_data / cls.__name__ / f"lambda_{laser_wavelength*1e9:.2f}nm.txt"

    @classmethod
    def _make_linelist_file(
        cls,
        laser_wavelength: float,
        state_initial: State = None,
        state_final: State = None,
    ) -> Transitions:
        raise NotImplementedError()

    @classmethod
    def get_linelist_file(cls, laser_wavelength: float = None) -> str:
        """Returns the path to the line list file.

        Returns:
            str: The path to the line list file.
        """
        raise NotImplementedError()

    @classmethod
    def get_all_transitions(
        cls, laser_wavelength: float = None, force_recalculate: bool = False
    ) -> Transitions:
        """Returns all possible transitions for the molecule.

        Returns:
            Transitions: The transitions.
        """

        if cls._has_linelist_file(laser_wavelength) and not force_recalculate:
            df: pd.DataFrame = parse_hitran_data(
                cls.get_linelist_file(laser_wavelength=laser_wavelength)
            )
            df = cls.process_hitran_data(df)

            return Transitions(df)
        else:
            print("Generating linelist file...")
            return cls._make_linelist_file(laser_wavelength)

    @classmethod
    def process_hitran_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Processes the HITRAN data.

        Args:
            df: The HITRAN data.

        Returns:
            pd.DataFrame: The processed data.
        """
        raise NotImplementedError()

    @classmethod
    def E(cls, state: State) -> float:
        return state.E

    @classmethod
    def dE(cls, transitions: Transitions) -> float:
        return transitions.vacuum_wavenumber

    @classmethod
    def degeneracy(cls, state) -> int:
        return state.degeneracy

    @classmethod
    def crosssection(cls, transitions: Transitions, lambda_laser: float) -> float:
        return transitions.crosssection

    @classmethod
    def get_intensity_variable(cls, transitions: Transitions, **temperatures) -> float:
        return cls.get_populations(transitions.state_initial, **temperatures)
