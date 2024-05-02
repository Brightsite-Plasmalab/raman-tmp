import pandas as pd
from ramlab.hitran.parser import parse_hitran_data
from ramlab.molecules.base import Molecule
from ramlab.molecules.transitions import Transitions


class LineListMolecule(Molecule):
    _transitions: Transitions = None

    def __init__(self):
        self._load_transitions()

    @classmethod
    @property
    def transitions(cls) -> Transitions:
        """Returns the transitions of the molecule.

        Returns:
            Transitions: The transitions.
        """
        if cls._transitions is None:
            cls._transitions = cls._load_transitions()
        return cls._transitions

    @classmethod
    def get_linelist_file(cls, laser_wavelength: float = None) -> str:
        """Returns the path to the line list file.

        Returns:
            str: The path to the line list file.
        """
        raise NotImplementedError()

    @classmethod
    def get_all_transitions(cls) -> Transitions:
        """Returns all possible transitions for the molecule.

        Returns:
            Transitions: The transitions.
        """
        return cls.transitions

    @classmethod
    def _load_transitions(cls) -> Transitions:
        df: pd.DataFrame = parse_hitran_data(cls.get_linelist_file())
        df = cls.process_hitran_data(df)

        cls._transitions = Transitions(df)
        return cls._transitions

    @classmethod
    def process_hitran_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Processes the HITRAN data.

        Args:
            df: The HITRAN data.

        Returns:
            pd.DataFrame: The processed data.
        """
        raise NotImplementedError()
