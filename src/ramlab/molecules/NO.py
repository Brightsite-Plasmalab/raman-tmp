
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
from sympy.physics.wigner import wigner_3j

class NO(AbInitioMolecule):
    def __init__(self):
        print("Loading NO molecule...")
        
        #self.states = self._get_all_transition_states()
       # inital,final = self.states
        #self.transitions = self._make_transitions(laser_wavelength= 532e-9,state_initial=inital, state_final=final)
        #self.transitions = self._get_all_transitions()
        #     self._make_transitions(laser_wavelength= 532e-9,state_initial=inital_state, state_final=final_state)
        
    # Molecule properties
    molecule_name = "NO"
    molecule_number = 8
    
    # Degeneracy constants
    g_e = 2  # nuclear degeneracy for even J - needs verification
    g_o = 2  # nuclear degeneracy for odd J - needs verification

    # Energy constants
    # For the electroic ground state: Omega = +/- 1/2
    w_e = 1904.20  # cm^-1 Vibrational constant at eq. Source:NIST
    w_ex_e = 14.075  # cm^-1 Vibrational anharmonicity at eq. Source:NIST
    B_e = 1.67195  # cm^-1 Rotational constant at eq. Source:NIST
    alpha0_e_1 = 0.0171  # cm^-1. Source: NIST

    # For the electronic higher state: Omega = +/- 3/2
    A = 123.160  # cm^-1. Spin-orbit coupling constant. Source: 
    w_e_high = 1904.04  # cm^-1 Vibrational constant at eq. Source:NIST
    w_ex_e_high = 14.075  # cm^-1 Vibrational anharmonicity at eq. Source:NIST
    B_e_high = 1.72016  # cm^-1 Rotational constant at eq. Source:NIST
    alpha0_e_1_high = 0.0182 # cm^-1. Source: NIST
    
    level_energies = {}

    @classmethod
    def B_v(cls, v) -> float:
        # See Derek A. Long, eq. 6.6.14
        return cls.B_e - cls.alpha0_e_1 * (v + 1 / 2)
    
    @classmethod
    def parity(cls, J, S):#Determine parity
       return (-1)**(J-S)    
    


    # Intensity and population calculations
    @classmethod
    def _calc_degeneracy(cls, transitions: Transitions):# Determine rotational degeneracy
        """Rotational degenarcy - needs actually validating... See Hougen 1970 or Lepard 1970"""
        J = transitions.J
        g = 2 *(J + 1)
        #g = (J * 0) +1
        return g
  
    @classmethod
    def _calc_crosssection(cls, transitions: Transitions):
        """ Cross sections - many terms need to be implemented"""
        b2 = cls.linestrength(transitions) #Plazcek-Teller Coeffecients, squared
        print ('Cross sections complete')
        #R2 = cls.R2(transitions) #Intensity correction factor for transitions in omega
        return b2#*R2

    @classmethod
    def _calc_depolarization_ratio(cls, transitions: Transitions):
        print ('Depolarization ratio complete')
        return 1.0

    @classmethod
    def _rc(cls,J, v, type): #Rotational coeffecients, used in calculation of linestrengths for Satija and Lucht method, taken from Zare - Angular Momentum, P303
        #Has been validatad against Fig. 3 from Satija and Lucht
        A = cls.A
        Bv = cls.B_v(v)
        #B_v = cls.B_v(v)
        Y = A/Bv #Ratio of spin-orbit coupling constant to rotational constant
        X = (4 * (J - 0.5) * (J + 1.5) + (Y - 2)**2)**0.5
        if type in ['a', 'd']:
            return ((X + (Y - 2)) / (2 * X))**0.5
        elif type in ['b', 'c']:
            coeff = ((X - (Y - 2)) / (2 * X))**0.5
            if type == 'c':
                coeff = -coeff
            return coeff
        else:
            raise ValueError("Invalid type. Type must be one of ['a', 'b', 'c', 'd'].")



    @classmethod
    # Testing new method based on Satija and Lucht
    def linestrength(cls, transitions: Transitions):
        rc = cls._rc
        intensities = []
        aq_0 = 1.0 
        aq_2 = 0.02   
        a_j = []
        b_j = []  
        J_list = [] 
        tran_length = len(transitions.linelist)
        print (f'Calculating linestrengths for {tran_length}...')
        for index, row in transitions.linelist.iterrows():
            O_l = row['lower_O']
            O_u = row['upper_O']
            J_l = row['lower_J']
            J_u = row['upper_J']
            v_l = row['lower_v']
            dS = row['dS']
            S_l = row['lower_S']
            S_u = row['upper_S']
            dO = row['dO']
            # Definition of F1 and F2 states are not yet correctly defined.
            # Attention also needs to be paid to the +/- term that preceeds the aq_2 term
            if ((O_l == 0.5) or (O_l == -0.5)) and (dO == 0) : # For F1 -> F1 pure rotational transitions - Eq. 40 from Satija and Lucht
                intensity = (2*J_l+1)*(2*J_u+1)/10 *( \
                aq_0 * ((rc(J_l,v_l,'a')*rc(J_u,v_l,'a')* wigner_3j(J_l, J_u, 2, 1/2, -1/2, 0)) - (rc(J_l,v_l,'b')*rc(J_u,v_l,'b') * wigner_3j(J_l, J_u, 2, 3/2, -3/2, 0)))  \
                + aq_2 * ((-1)**(J_l - 0.5)) * (-(rc(J_l,v_l,'a')*rc(J_u,v_l,'b')* wigner_3j(J_l, J_u, 2, -1/2, -3/2, 2)) + (rc(J_l,v_l,'b')*rc(J_u,v_l,'a') * wigner_3j(J_l, J_u, 2, -3/2, -1/2, 2)))\
                )**2
            elif ((O_l == 1.5) or (O_l == -1.5)) and (dO == 0) :# For F2 -> F2 pure rotational transitions - Eq. 41 from Satija and Lucht
                intensity = (2*J_l+1)*(2*J_u+1)/10 *( \
                aq_0 * ((rc(J_l,v_l,'c')*rc(J_u,v_l,'c')* wigner_3j(J_l, J_u, 2, 1/2, -1/2, 0)) - (rc(J_l,v_l,'d')*rc(J_u,v_l,'d') * wigner_3j(J_l, J_u, 2, 3/2, -3/2, 0)))  \
                + aq_2 * ((-1)**(J_l - 0.5)) * (-(rc(J_l,v_l,'c')*rc(J_u,v_l,'d')* wigner_3j(J_l, J_u, 2, -1/2, -3/2, 2)) + (rc(J_l,v_l,'d')*rc(J_u,v_l,'c') * wigner_3j(J_l, J_u, 2, -3/2, -1/2, 2))) \
                )**2
            #For F1 -> F2 electronic-rotational transitions - Eq. 60 from Satija and Lucht
            #elif (dO == 1) or (dO == -1): #Note place holder for F1 -> F2 transitions, equation is correctly implemented but rules need to be defined
            elif (np.isin(dO, [1,-1])):
                intensity = (2*J_l+1)*(2*J_u+1)/5 *( \
                aq_0 * ((rc(J_l,v_l,'a')*rc(J_u,v_l,'c')* wigner_3j(J_l, J_u, 2, 1/2, -1/2, 0)) - (rc(J_l,v_l,'b')*rc(J_u,v_l,'d') * wigner_3j(J_l, J_u, 2, 3/2, -3/2, 0)))  \
                + aq_2 * ((-1)**(J_l - 0.5)) * (-(rc(J_l,v_l,'a')*rc(J_u,v_l,'d')* wigner_3j(J_l, J_u, 2, -1/2, -3/2, 2)) + (rc(J_l,v_l,'b')*rc(J_u,v_l,'c') * wigner_3j(J_l, J_u, 2, -3/2, -1/2, 2)))\
                )**2
            else: print(f'No valid transition for: Omega_lower = {O_l} and delta_Omega = {dO}')
            a_j.append(rc(J_l,v_l,'a'))
            b_j.append(rc(J_l,v_l,'b'))
            J_list.append(J_l)
            intensities.append(float(intensity / ((2*J_l)+1)))
        # plt.figure()
        # plt.scatter(J_list, a_j)
        # plt.scatter(J_list, b_j)
        # plt.show()
        return intensities

    #Energy calculations
    @classmethod
    def E(cls, state:State):#Perhaps check if this is unessary duplication?
        E_r = cls.E_rot(state)
        E_v = cls.E_vib(state)
        E = E_r + E_v
        #state.E = {'value': E, 'units': 'cm⁻¹'} 
        #print(E)
        return E

    # @classmethod
    # def E_rot(cls, state: State) -> float:
    #     """Calculates and stores energy levels (in units: cm^-1) including perturbations for each J.
    #     The matrix M represents the Hamiltonian including perturbations."""
    #     j, o = state.J, state.O
    #     E_r = np.zeros(len(j))
        
    #     # Construct the Hamiltonian matrix with perturbation
    #     for idx, (J, O) in enumerate(zip(j, o)):
    #         E11 = cls.B_e * (J - 1/2) * (J + 3/2)
    #         E12 = np.sqrt(cls.B_e * (J - 1/2) * (J + 3/2))
    #         E21 = E12
    #         E22 = (cls.A - 2 * cls.B_e) + cls.B_e * (J - 1/2) * (J + 3/2)
    #         M = np.array([[E11, E12], [E21, E22]])
    #         # Diagonalize the Hamiltonian to find perturbed energies
    #         eigenvalues = np.linalg.eigvals(M) #eigenvalues are energy levels
    #         if np.abs(O) == 0.5:
    #             E_r[idx] = eigenvalues[0]
    #         elif np.abs(O) == 1.5: 
    #             E_r[idx] = eigenvalues[1] 
    #         else:
    #             print(f"Invalid Omega value: {O}")
    #             raise ValueError("Invalid Omega value")     
    #     #print(E_r)
    #     return E_r
                 
    @classmethod
    def E_rot(cls, state: State) -> float:
        """Calculates and stores energy levels (in units: cm^-1) including perturbations for each J.
        The matrix M represents the Hamiltonian including perturbations."""
        J, p, v = state.J, state.p, state.v
        A = cls.A
        Bv = cls.B_v(v)
        Y = A/Bv #Ratio of spin-orbit coupling constant to rotational constant
        X = (4 * (J - 0.5) * (J + 1.5) + (Y - 2)**2)**0.5
        E_r = Bv * (((J - 0.5) * (J + 1.5)) + X/2)
        return E_r
    
    @classmethod
    def E_vib(cls, state: State) -> float: #Needs to be implemented, 0 set as placeholder
        v = state.v
        # See Derek A. Long, eq. 5.9.3
        E_v = (v + 1 / 2) * cls.w_e - cls.w_ex_e * (v + 1 / 2) ** 2
        return E_v
    
    #State calculations
    @classmethod
    def _get_all_transition_states(cls) -> tuple[State, State]:
    # Quantum numbers definitions - see Zare - Angular Momentum, P297
        vi = np.arange(0, 1)  # Vibrational 
        Ri = np.arange(0, 30, 1)  # Nuclear rotational angular momentum
        Si = np.array([0.5, -0.5])    # Electronic spin angular momentum
        Li = np.array([1, -1])   #  Electronic orbital angular momentum 

        # Generate all possible combinations of these quantum numbers
        V, R, S, L = np.meshgrid(vi, Ri, Si, Li, indexing='ij')
        V, R, S, L = V.flatten(), R.flatten(), S.flatten(), L.flatten()
        # Calculate other Quantum numbers
        O = np.abs(L + S ) # Calculate Omega for each state
        J = np.abs(R + S + L) # Total angular momentum
        N = J - S  # Determine quantum number N

        # Define allowed changes (dv, dJ, dO) 
        dv = np.array([ 0])  # Allowed change in vibrational quantum number
        dJ = np.array([-2, -1, 0, 1, 2])  # Allowed change in total angular momentum
        #dL = np.array([-2, 0, 2])  # Allowed change in electronic orbital angular momentum
        dS = np.array([-1,0,1])  # Allowed change in electronic spin angular momentum

        # Calculate transitions - first remake list of inital states, having the same length as the final states list 
        shape = (len(V), len(dv), len(dJ), len(dS))
        VI = np.broadcast_to(V[:, None, None, None], shape) 
        RI = np.broadcast_to(R[:, None, None, None], shape) 
        SI = np.broadcast_to(S[:, None, None, None], shape)
        LI = np.broadcast_to(L[:, None, None, None], shape)
        JI = RI + SI + LI # Total angular momentum
        OI = LI + SI  # Calculate Omega for each state: +3/2, +1/2, -1/2, -3/2
        NI = JI - SI # Determine quantum number N
        PI = cls.parity(JI, SI) #Calculate parity

        VF = VI + dv[:, None, None]
        JF = JI + dJ[:, None]
        SF = SI + dS
        LF = LI# + dL
        OF = LF + SF
        NF = JF - SF
        RF = NF - LF
        PF = cls.parity(JF, SF) 

        # Flatten the arrays to create a list of final states
        vi, ri, ji, si, li, oi, ni, pi = VI.flatten(), RI.flatten(), JI.flatten(), SI.flatten(), LI.flatten(), OI.flatten(), NI.flatten(), PI.flatten()
        vf, rf, jf, sf, lf, of, nf, pf = VF.flatten(), RF.flatten(), JF.flatten(), SF.flatten(), LF.flatten(), OF.flatten(), NF.flatten(), PF.flatten()

        # Apply validity conditions
        legal = (vf >= 0)                 #Final v and J states are positive
        legal = legal & (jf >= 0.5) & (ji >= 0.5)      # J must be at least 1/2
        legal = legal & (-1.5 <= of) & (of <= 1.5 )     # Omega is between -3/2 and +3/2
        legal = legal &  np.isin(sf, [-1/2, 1/2])       # Final spin is either -1/2 or +1/2
        #legal = legal & (np.abs(nf-ni) <=2)            # Delta N is 1 or 0
        #legal = legal & (pi == pf)                      # Parity must be conserved, i.e. + -> + or - -> -
        for i in [vf[legal],jf[legal],sf[legal],lf[legal],of[legal]]:print(np.unique(i))
        states_inital= State(v=vi[legal], R=ri[legal], S=si[legal], L=li[legal], J=ji[legal], O=oi[legal], N=ni[legal], p = pi[legal])
        states_final = State(v=vf[legal], R=rf[legal], S=sf[legal], L=lf[legal], J=jf[legal], O=of[legal], N=nf[legal], p = pf[legal])
        print(len(jf[legal]))
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


#no = NO
# print("getting transitions...")
#transitions = NO.get_all_transitions(laser_wavelength=532.5e-9, force_recalculate=True)
# print("Finito")
# transitions = transitions.sortby("vacuum_wavenumber", ascending=False)[:100] 
# print(transitions)


# no._get_all_states()
# #Example of accessing state data
# # print (no.states[1])
# # transitions = NO._make_transitions(laser_wavelength= 532e-9,state_initial = State(v=0, J=0, O=0.5, S=0.5, L=1), state_final = State(v=1, J=1, O=1.5, S=0.5, L=2)  )  
# # print('transitions complete')
# # print(transitions.vacuum_wavenumber)

# #Example of accessing transition data