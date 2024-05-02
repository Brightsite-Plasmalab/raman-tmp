import pandas as pd
from ramlab.molecules.base import Molecule
from ramlab.dirs import dir_data
from ramlab.hitran.parser import parse_hitran_data
from ramlab.molecules.hitran_linelist_molecule import LineListMolecule
from ramlab.molecules.state import State
import numpy as np
from scipy.constants import k, h, c

from ramlab.molecules.transitions import Transitions


class CH4(LineListMolecule):
    @classmethod
    def process_hitran_data(cls, df2: pd.DataFrame) -> pd.DataFrame:
        ## Parse data specific to the methane molecule
        # Upper global quanta
        df2["upper_quanta"] = df2["upper_quanta_global"] + df2["upper_quanta_local"]
        df2["upper_v1"] = df2["upper_quanta_global"].str[3:5].astype("Int64")
        df2["upper_v2"] = df2["upper_quanta_global"].str[5:7].astype("Int64")
        df2["upper_v3"] = df2["upper_quanta_global"].str[7:9].astype("Int64")
        df2["upper_v4"] = df2["upper_quanta_global"].str[9:11].astype("Int64")
        df2["upper_mp"] = df2["upper_quanta_global"].str[11:13].astype("Int64")
        df2["upper_sy"] = df2["upper_quanta_global"].str[13:15]

        # Upper local quanta
        # See doi:10.1016/S0022-4073(03)00155-9 for meaning
        # Made some adjustments in the format as it appears they did not strictly follow the HITRAN2004 format:
        #         I3 1X A2 1X I3 A5
        # HITRAN: 2X I3 A2    I3 A5
        df2["upper_J"] = df2["upper_quanta_local"].str[0:3].astype("Int64")
        df2["upper_C"] = df2["upper_quanta_local"].str[4:6]
        df2["upper_a"] = df2["upper_quanta_local"].str[7:10].astype("Int64")
        df2["upper_F"] = df2["upper_quanta_local"].str[10:15]

        # Lower global quanta
        df2["lower_quanta"] = df2["lower_quanta_global"] + df2["lower_quanta_local"]
        df2["lower_v1"] = df2["lower_quanta_global"].str[3:5].astype("Int64")
        df2["lower_v2"] = df2["lower_quanta_global"].str[5:7].astype("Int64")
        df2["lower_v3"] = df2["lower_quanta_global"].str[7:9].astype("Int64")
        df2["lower_v4"] = df2["lower_quanta_global"].str[9:11].astype("Int64")
        df2["lower_mp"] = df2["lower_quanta_global"].str[11:13].astype("Int64")
        df2["lower_sy"] = df2["lower_quanta_global"].str[13:15]

        # Lower local quanta
        df2["lower_J"] = df2["lower_quanta_local"].str[0:3].astype("Int64")
        df2["lower_C"] = df2["lower_quanta_local"].str[4:6]
        df2["lower_a"] = df2["lower_quanta_local"].str[7:10].astype("Int64")
        df2["lower_F"] = df2["lower_quanta_local"].str[10:15]

        # Filter for the ground rotational state
        idx_ground_rotational_state = df2["lower_J"] == 0

        # Map the ground state transitions to the ground state energy
        ground_state_transitions = pd.DataFrame(
            index=df2["lower_quanta_global"][idx_ground_rotational_state].values,
            data=df2["lower_E"][idx_ground_rotational_state].values,
            columns=["lower_E"],
        ).drop_duplicates()

        # For each transition, get the energy of the ground state
        idx_global_quanta = df2["lower_quanta_global"]

        # Add the energy of the vibrational state
        df2["lower_E_vib"] = ground_state_transitions.loc[idx_global_quanta][
            "lower_E"
        ].values
        df2["lower_E_rot"] = df2["lower_E"] - df2["lower_E_vib"]

        # TODO: Contact maintainers of MeCaSDa to confirm the einstein coefficient is the cross-section.
        # The "room temperature intensity" divided by the boltzmann distribution and degeneracies results in a constant relation with the Einstein coefficient.
        # So I suspect that the Einstein coefficient is the cross-section.
        df2["crosssection"] = df2["einstein_A_coefficient"].values
        # df2["crosssection"] = df2["intensity"].values / np.exp(
        #     -100 * h * c * df2["lower_E"].values / (k * 296)
        # )

        return df2

    @classmethod
    def get_linelist_file(cls, laser_wavelength: float = None) -> str:
        return dir_data / "mecasda" / "extract_12CH4_2850_3000_0_pol.txt"

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
    def get_populations(cls, state_initial, **temperatures) -> float:
        # Assume thermal equilibrium for now
        if len(temperatures) != 1:
            raise ValueError(
                "Only one temperature is allowed for the CH4 molecule for now. Fitting of Tr=/=Tv is not yet implemented."
            )

        T = temperatures["T"]
        g = state_initial.degeneracy
        E = state_initial.E
        weights = g * np.exp(-100 * h * c * E / (k * T))
        sumofstates = np.sum(weights)
        n = weights / sumofstates
        return n

    @classmethod
    def get_intensity_variable(cls, transitions: Transitions, **temperatures) -> float:
        return cls.get_populations(transitions.state_initial, **temperatures)


if __name__ == "__main__":
    M = CH4()
