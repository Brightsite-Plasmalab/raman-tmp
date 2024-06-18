
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from itertools import product
# # Add the src directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ramlab.molecules.state import State
from ramlab.molecules.transitions import Transitions
from ramlab.molecules.ab_initio_molecule import AbInitioMolecule


class NO(AbInitioMolecule):
    def __init__(self):
        print("Loading NO molecule...")
        self.states = self._get_all_transition_states()
        inital,final = self.states
        self.transitions = self._make_transitions(laser_wavelength= 532e-9,state_initial=inital, state_final=final)
        print("Calculation complete")
        #self.transitions = self._get_all_transitions()
        #     self._make_transitions(laser_wavelength= 532e-9,state_initial=inital_state, state_final=final_state)
        
    # Molecule properties
    molecule_name = "NO"
    molecule_number = 8
    
    # Degeneracy constants
    g_e = 2  # nuclear degeneracy for even J - needs verification
    g_o = 2  # nuclear degeneracy for odd J - needs verification

    # Energy constants
    w_e = 1904.9  # cm^-1 Vibrational constant at eq. Source:NIST
    w_ex_e = 14.075  # cm^-1 Vibrational anharmonicity at eq. Source:NIST
    B_e = 1.67195  # cm^-1 Rotational constant at eq. Source:NIST

    A = 123.160  # cm^-1. Spin-orbit coupling constant. Source: 
    level_energies = {}

    @classmethod
    def parity(self, J,S = 1/2):#Determine parity
        if not ((J * 2) % 1 == 0 and (J * 2) % 2 == 1):
            raise ValueError("J for NO must be a half-integer value.")
        elif J% 0.5 ==0:
            return (-1)**(J-S)    
    
    # Intensity and population calculations
    @classmethod
    def _calc_degeneracy(cls, transitions: Transitions):# Determine rotational degeneracy
        """Rotational degenarcy - needs actually validating... See Hougen 1970 or Lepard 1970"""
        J = transitions.J
        g = 2 * (J + 1)
        return g
  

    @classmethod
    def _calc_crosssection(cls, transitions: Transitions):
        b2 = cls.PT_coeffecients(transitions) #Plazcek-Teller Coeffecients, squared
        #gamma = cls.gamma
        return b2#*gamma

    @classmethod
    def _calc_depolarization_ratio(cls, transitions: Transitions):
        return 1

    @classmethod
    def PT_coeffecients(cls, transitions: Transitions) -> float:# Plazcek-Teller Coeffecients
        """Plazcek-Teller Coeffecients, describes the probability of certian rotational transition occurring - function of J, Omega, DeltaJ, and DeltaOmega"""    
        dJ = transitions.dJ
        dO = transitions.dO
        J = transitions.state_initial.J
        Omega = transitions.state_initial.O
        b = 0
        b_values = []
        for dJ, dO, J, Omega in product(dJ, dO, J, Omega):
            
            # Delta Omega = 0
            if (dJ ==0) & (dO == 0): b = (3*Omega**2 - J*(J+1)) / np.sqrt (J*(J+1)*(2*J-1)*(2*J+3))
            if (dJ == 1) & (dO == 0): b = Omega * np.sqrt(3*((J+1)**2 - Omega**2) / (J*(J+1)*(2*J+1)*(2*J+3)))
            if (dJ == -1) & (dO == 0): b = Omega * np.sqrt(3*(J**2 - Omega**2) / (J*(J-1)*(2*J+1)*(2*J-1)))
            if (dJ == 2) & (dO == 0): b =np.sqrt(3*((J+1)**2 - Omega**2)*((J+2)**2 - Omega**2) / (2*(J+1)*(J+2)*(2*J+1)*(2*J+3)))
            if (dJ == -1) & (dO == 0): b = np.sqrt(3*(J**2 - Omega**2)*((J-1)**2 - Omega**2) / (2*J*(J-1)*(2*J-1)*(2*J+1)))
            
            # Delta Omega = + 1
            if (dJ ==0) & (dO == 1): b = (2*Omega + 1) * np.sqrt((3*(J + Omega)*(J + Omega + 1)) / (2*J*(J + 1)*(2*J - 1)*(2*J + 3)))
            if (dJ ==1) & (dO == 1): b = (J-2*Omega) *np.sqrt(((J+Omega+1)*(J+Omega+2))/(2*J*(J+1)*(J+2)*(2*J+1)))
            if (dJ ==-1) & (dO == 1): b = (J + 2*Omega + 1) * np.sqrt(((J + Omega) * (J + Omega - 1)) / (2 * J * (J - 1) * (J + 1) * (2 * J + 1)))
            if (dJ ==2) & (dO == 1): b = -np.sqrt(((J + 1)**2 - Omega**2) * (J + Omega + 2) * (J + Omega + 3) / ((J + 1) * (J + 2) * (2*J + 1) * (2*J + 3)))
            if (dJ ==-2) & (dO == 1): b = np.sqrt(((J**2 - Omega**2) * (J + Omega - 1) * (J + Omega - 2)) / (J * (J - 1) * (2*J - 1) * (2*J + 1)))
        
            #Delta Omega = -1
            if (dJ ==0) & (dO == -1): b = (2*Omega - 1) * np.sqrt((3*(J - Omega)*(J - Omega + 1)) / (2*J*(J + 1)*(2*J - 1)*(2*J + 3)))
            if (dJ ==1) & (dO == -1): b =(J + 2*Omega) * np.sqrt(((J - Omega + 1) * (J - Omega + 2)) / (2 * J * (J + 1) * (J + 2) * (2 * J + 1)))
            if (dJ ==-1) & (dO == -1):b=(J - 2*Omega + 1) * np.sqrt(((J - Omega) * (J - Omega - 1)) / (2 * J * (J - 1) * (J + 1) * (2 * J + 1)))
            if (dJ ==+2) & (dO == -1): b=np.sqrt(((J + 1)**2 - Omega**2) * (J - Omega + 2) * (J - Omega + 3) / ((J + 1) * (J + 2) * (2*J + 1) * (2*J + 3)))
            if (dJ ==-2) & (dO == -1):b= -np.sqrt(((J**2 - Omega**2) * (J - Omega - 1) * (J - Omega - 2)) / (J * (J - 1) * (2*J - 1) * (2*J + 1)))
        
            # Delta Omega = + 2
            if (dJ == 0) & (dO == 2): b = np.sqrt((3 * (J + Omega + 1) * (J + Omega + 2) * (J - Omega)) / (2 * J * (J + 1) * (2 * J - 1) * (2 * J + 3)))
            if (dJ == 1) & (dO == 2): b = -np.sqrt((J + Omega) * (J + Omega + 1) * (J + Omega + 2) * (J + Omega + 3) / (2 * J * (J + 1) * (J + 2) * (2 * J + 1)))
            if (dJ == -1) & (dO == 2): b = np.sqrt((J + Omega + 1) * (J + Omega) * (J + Omega - 1) * (J + Omega - 2) / (2 * J * (J - 1) * (J + 1) * (2 * J + 1)))
            if (dJ == 2) & (dO == 2): b = np.sqrt((J + Omega + 1) * (J + Omega + 2) * (J + Omega + 3) * (J + Omega + 4) / (4 * J * (J + 1) * (J + 2) * (2 * J + 1) * (2 * J + 3)))
            if (dJ == -2) & (dO == 2): b = np.sqrt((J + Omega) * (J + Omega - 1) * (J + Omega - 2) * (J + Omega - 3) / (4 * J * (J - 1) * (2 * J - 1) * (2 * J + 1)))
            
            # Delta Omega = -2
            if (dJ == 0) & (dO == -2): b = np.sqrt((3 * (J - Omega + 1) * (J - Omega + 2) * (J + Omega)) / (2 * J * (J + 1) * (2 * J - 1) * (2 * J + 3)))
            if (dJ == 1) & (dO == -2): b = np.sqrt((J - Omega) * (J - Omega + 1) * (J - Omega + 2) * (J - Omega + 3) / (2 * J * (J + 1) * (J + 2) * (2 * J + 1)))
            if (dJ == -1) & (dO == -2): b = -np.sqrt((J - Omega + 1) * (J - Omega) * (J - Omega - 1) * (J - Omega - 2) / (2 * J * (J - 1) * (J + 1) * (2 * J + 1)))
            if (dJ == 2) & (dO == -2): b = np.sqrt((J - Omega + 1) * (J - Omega + 2) * (J - Omega + 3) * (J - Omega + 4) / (4 * J * (J + 1) * (J + 2) * (2 * J + 1) * (2 * J + 3)))
            if (dJ == -2) & (dO == -2): b = np.sqrt((J - Omega) * (J - Omega - 1) * (J - Omega - 2) * (J - Omega - 3) / (4 * J * (J - 1) * (2 * J - 1) * (2 * J + 1)))
            
            b_values.append(b)
        print (b_values)
        return b**2
    
    def gamma(cls, transitions: Transitions): #Intensity correction factor for transitions in lambda
        """Really needs checking..."""
        dO = transitions.dO 
        if dO == 0:
            y = 1
        if dO == 1 or -1:
            y = 0.1       
        return y

    #Energy calculations
    @classmethod
    def E(cls, state:State):#Perhaps check if this is unessary duplication?
        E_r = cls.E_rot(state)
        E_v = cls.E_vib(state)
        E = E_r + E_v
        #state.E = {'value': E, 'units': 'cm⁻¹'} 
        #print(E)
        return E

    @classmethod
    def E_rot(cls, state: State) -> float:
        """Calculates and stores energy levels (in units: cm^-1) including perturbations for each J.
        The matrix M represents the Hamiltonian including perturbations."""
        j, o = state.J, state.O
        E_r = np.zeros(len(j))
        
        # Construct the Hamiltonian matrix with perturbation
        for idx, (J, O) in enumerate(zip(j, o)):
            E11 = cls.B_e * (J - 1/2) * (J + 3/2)
            E12 = np.sqrt(cls.B_e * (J - 1/2) * (J + 3/2))
            E21 = E12
            E22 = (cls.A - 2 * cls.B_e) + cls.B_e * (J - 1/2) * (J + 3/2)
            M = np.array([[E11, E12], [E21, E22]])
            # Diagonalize the Hamiltonian to find perturbed energies
            eigenvalues = np.linalg.eigvals(M) #eigenvalues are energy levels
            if np.abs(O) == 0.5:
                E_r[idx] = eigenvalues[0]
            elif np.abs(O) == 1.5: 
                E_r[idx] = eigenvalues[1] 
            else:
                print(f"Invalid Omega value: {O}")
                raise ValueError("Invalid Omega value")     
        #print(E_r)
        return E_r
                 
    @classmethod
    def E_vib(cls, state: State) -> float: #Needs to be implemented, 0 set as placeholder
        # E_v = 0  
        # return E_v
        v = state.v
        # See Derek A. Long, eq. 5.9.3
        E_v = (v + 1 / 2) * cls.w_e - cls.w_ex_e * (v + 1 / 2) ** 2
        return E_v
    
    # State calculations
    @classmethod
    def _get_all_transition_states(cls) -> tuple[State, State]:
        """Calculates all possible combinations of inital and final states for transitions and applies selection rules"""
        # Quantum numbers definitions
        vi = np.arange(0, 8)  # Vibrational state
        Ji = np.arange(0.5, 31.5, 1)  # Total angular momentum, half-integer
        Si = np.array([0.5, -0.5])    # Projection of the electronic spin on the internuclear axis
        Li = np.array([1, -1])       #  Projection of the electronic angular momentum on the internuclear axis   
        # Generate all possible combinations of these quantum numbers
        V, J, S, L = np.meshgrid(vi, Ji, Si, Li, indexing='ij')
        V, J, S, L = V.flatten(), J.flatten(), S.flatten(), L.flatten()
        O = L + S  # Calculate Omega for each state
        R = J - O # Calculate R for each state
        valid_inital = (R >= 0)  # Filter invalid states
        
        # Define allowed changes (dv, dJ, dO)
        dv = np.array([0])  # Allowed change in vibrational quantum number
        dJ = np.array([-2, -1, 0, 1, 2])  # Allowed change in total angular momentum
        dO = np.array([-1, 0, 1])  # Allowed change in Omega
        dS = np.array([0])  # Allowed change in spin
        dL = dO # Allowed change in L

        # Calculate transitions - first remake list of inital states, having the same length as the final states list 
        shape = (len(V[valid_inital]), len(dv), len(dJ), len(dO))
        VI = np.broadcast_to(V[valid_inital][:, None, None, None], shape) 
        JI = np.broadcast_to(J[valid_inital][:, None, None, None], shape) 
        SI = np.broadcast_to(S[valid_inital][:, None, None, None], shape)
        LI = np.broadcast_to(L[valid_inital][:, None, None, None], shape)
        OF = LI + SI

        VF = VI + dv[:, None, None]
        JF = JI + dJ[:, None]
        SF = SI + dS
        LF = LI + dL
        OF = LF + SF
        # Flatten the arrays to create a list of final states
        vi, ji, si, li, oi = VI.flatten(), JI.flatten(), SI.flatten(), LI.flatten(), OF.flatten()
        
        vf, jf, sf, lf, of = VF.flatten(), JF.flatten(), SF.flatten(), LF.flatten(), OF.flatten()
        
        # Apply validity conditions
        valid_final = (vf >= 0) & (jf >= 0.5) & (np.mod(jf, 1) == 0.5) & (-1.5 <= of) & (of <= 1.5 )
        states_inital= State(v=vi[valid_final], J=ji[valid_final], O=oi[valid_final], S=si[valid_final], L=li[valid_final] )
        states_final = State(v=vf[valid_final], J=jf[valid_final], O=of[valid_final], S=sf[valid_final], L=lf[valid_final])
        
        return states_inital, states_final    

    @classmethod
    def _format_quanta_global(cls, state: State):
        return np.char.mod("%2d", state.v)
    

    @classmethod
    def _format_quanta_local(cls, state: State):
        return np.char.mod("%2d", state.J)
        

    @classmethod
    def process_hitran_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        # Parse the quantum numbers from the HITRAN format for local/global quantum numbers
        # e.g.
        #   df["lower_v"] = df["lower_quanta_global"].str[0:2].astype("Int64")
        raise NotImplementedError()

    @classmethod
    def _validate_transitions(cls, transitions: Transitions):
        return super()._validate_transitions(transitions)



no = NO
print("getting transitions...")
transitions = no.get_all_transitions(laser_wavelength=532.5e-9, force_recalculate=True)
print("Finito")
transitions = transitions.sortby("vacuum_wavenumber", ascending=False)[:100] 
print(transitions)


# no._get_all_states()
# #Example of accessing state data
# # print (no.states[1])
# # transitions = NO._make_transitions(laser_wavelength= 532e-9,state_initial = State(v=0, J=0, O=0.5, S=0.5, L=1), state_final = State(v=1, J=1, O=1.5, S=0.5, L=2)  )  
# # print('transitions complete')
# # print(transitions.vacuum_wavenumber)

# #Example of accessing transition data
