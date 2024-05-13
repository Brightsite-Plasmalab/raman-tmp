from ramlab.molecules.diatomic import SimpleDiatomicMolecule
from ramlab.molecules.state import State
from ramlab.molecules.transitions import Transitions
import numpy as np


class H2(SimpleDiatomicMolecule):
    # Molecule properties
    molecule_number = 45
    molecule_name = "H2"
    isotope_number = 1

    # Degeneracy constants
    g_e = 1  # nuclear degeneracy for even J
    g_o = 3  # nuclear degeneracy for odd J

    # Energy constants
    w_e = 4.39449e03  # /cm
    w_ex_e = 1.16986e02  # /cm
    B_e = 6.07823e01  # /cm
    alpha0_e_1 = 2.94099e00  # /cm
    D1_e = 4.66150e-02  # /cm
    alpha1_e_1 = 2.37251e-03  # /cm
    D2_e = 5.08134e-05  # /cm
    alpha2_e_1 = 1.35508e-05  # /cm
    D3_e = 0
    alpha3_e_1 = 0

    @classmethod
    def _validate_transitions(cls, transitions: Transitions):
        transitions = super()._validate_transitions(transitions)

        idx = np.isin(transitions.dv, [-1, 0, 1]) & np.isin(transitions.dJ, [-2, 0, 2])

        return transitions[idx]

    @classmethod
    def _calc_degeneracy(cls, state: State):
        degeneracy_nuclear = (state.J % 2) * cls.g_o + ((state.J + 1) % 2) * cls.g_e
        degeneracy = (2 * state.J + 1) * degeneracy_nuclear
        return degeneracy

    @classmethod
    def _calc_crosssection(cls, transitions: Transitions):
        return 1

    @classmethod
    def _calc_depolarization_ratio(cls, transitions: Transitions):
        return 1
