from ramlab.molecules.state import State
from ramlab.molecules.transitions import Transitions
import numpy as np
from scipy.constants import k, h, c


# TODO: Communicate units clearly
class Molecule:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "The Molecule class should not be used as an instance. Did you use something like `M=CH4()`? Replace it with `M=CH4`"
        )

    @classmethod
    def E(cls, state: State) -> float:
        """Returns the energy of a given state.

        Args:
            state: The quantum numbers of the state.

        Returns:
            float: The energy of the state in cm^-1.
        """
        raise NotImplementedError()

    @classmethod
    def dE(cls, transitions: Transitions) -> float:
        """Returns the energy difference between two states.

        Args:
            state_initial: The initial state.
            state_final: The final state.

        Returns:
            float: The energy difference between the two states in cm^-1.
        """
        return cls.E(transitions.state_final) - cls.E(transitions.state_initial)

    @classmethod
    def degeneracy(state: State) -> int:
        """Returns the degeneracy of a given state.

        Args:
            **state: The quantum numbers of the state.

        Returns:
            int: The degeneracy of the state.
        """
        raise NotImplementedError()

    @classmethod
    def crosssection(cls, transitions: Transitions, lambda_laser: float) -> float:
        """Returns the cross section of a transition between two states.

        Args:
            state_initial: The initial state.
            state_final: The final state.
            lambda_laser: The wavelength of the laser in meters.

        Returns:
            float: The cross section of the transition in m^2.
        """
        raise NotImplementedError()

    @classmethod
    def depolarization_ratio(cls, transitions: Transitions) -> float:
        """Returns the depolarization ratio of a transition between two states.

        Args:
            state_initial: The initial state.
            state_final: The final state.

        Returns:
            float: The depolarization ratio of the transition.
        """
        raise NotImplementedError()

    @classmethod
    def get_all_transitions(cls, laser_wavelength: float = None) -> Transitions:
        """Returns all possible transitions for the molecule.

        Returns:
            Transitions: The transitions.
        """
        raise NotImplementedError()

    @classmethod
    def get_populations(cls, state_initial, **temperatures) -> float:
        """Returns the populations of a state.

        Args:
            state_initial: The initial state.

        Returns:
            float: The populations of the state.
        """

        # Assume thermal equilibrium for now
        if len(temperatures) != 1:
            raise ValueError(
                f"Only one temperature is allowed for the base molecule. Fitting of Tr=/=Tv is not yet implemented for {cls.__name__}."
            )

        T = temperatures["T"]
        g = state_initial.degeneracy
        E = state_initial.E
        weights = g * np.exp(-100 * h * c * E / (k * T))
        sumofstates = np.sum(weights)
        n = weights / sumofstates
        return n

    @classmethod
    def get_intensity_constant(
        cls, transitions: Transitions, laser_wavelength: float
    ) -> float:
        """Returns the constant part of the intensity calculation. These involve physical constants and cross-sections, but not the populations.

        Args:
            transitions: The transitions.
            laser_wavelength: The wavelength of the laser in meters.

        Returns:
            float: The intensity constant of the transition.
        """
        return cls.crosssection(transitions, laser_wavelength)

    @classmethod
    def get_intensity_variable(cls, transitions: Transitions, **temperatures) -> float:
        """Returns the variable part of the intensity calculation. These involve populations, but not physical constants or cross-sections.
        When fitting temperatures, this is the only part that changes. Calculating this separately greatly speeds up the fitting process.

        Args:
            transitions: The transitions.

        Returns:
            float: The intensity variable of the transition.
        """
        raise NotImplementedError()

    @classmethod
    def get_intensity(
        cls, transitions: Transitions, laser_wavelength=None, **temperatures
    ) -> float:
        """Returns the intensity of a transition.

        Args:
            transitions: The transitions.

        Returns:
            float: The intensity of the transition.
        """
        I_c = cls.get_intensity_constant(transitions, laser_wavelength)
        I_v = cls.get_intensity_variable(transitions, **temperatures)
        return I_c * I_v
