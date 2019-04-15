"""
qdsystem:
Package for simulating quantum dot systems with or without leads.

Work flow of a qdsystem calculation:
    1. Initialize an instance of the qdsystem class.
    2. Add quantum dots with their corresponding charging energies and other properties with self.add_dot()
    3. Add tunnel couplings between dots with self.add_coupling().
    4. Add leads to dots of your choice with self.add_lead().
    5. Port the system to kwant for a given maximum charge per dot (if
       there are leads) or total charge of the system (if there are no leads) N using self.to_kwant(N)
    6. Calculate properties of the system using additional functionality to come.
"""

### TO DO:
### ADD DELETE_DOT FUNCTION

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import warnings
import time

from itertools import product, combinations, takewhile
from sympy.utilities.iterables import multiset_permutations

class DotSystem:
    """
    Class for simulating systems of multiple quantum dots.

    Attributes:
    dots: Sorted dictionary of properties for the system's quantum dots
    couplings: Sorted dictionary capacitances/couplings between dots
    leads: Sorted dictionary of leads and their properties
    sys: kwant system for porting system properties to kwant
    print: whether or not to print results of calling methods
    ndots: number of dots in system
    ncouplings: number of couplings between dots in system
    nleads: number of leads in system
    ndotscharge: number of charge type dots
    ndotsfermionic: number of fermionic type dots
    ndotssc: number of superconducting type dots
    _quasidots: 
    """
    def __init__(self, verbose = True):
        """
        Constructor for DotSystem class.

        Parameters:
        print (bool): whether or not to print results of calling methods
        """
        self.dots = {}          # Dictionary of dots & their parameters
        self.couplings = {}     # Dictionary of couplings/capacitances between dots
        self.leads = {}         # Dictionary of leads and which dots they attach to
        self.sys = []           # List of dictionaries of state info for each quantum state of system.
        self._indices = []      # Dictionary which provides index for each dot/lead name, ordering them.
        self.verbose = verbose  # Suppresses printing function results if False
    
    def __str__(self):
        """Print class type and attributes when qdsystem is called with print()."""
        output = "qdsystem class object with: \n"
        output += str(self.ndots) + " total dots, with "
        output += str(self.nleads) + " leads."
        return output

    @property
    def ndots(self):
        """Number of dots in system."""
        return len(self.dots)

    @property
    def ndotseff(self):
        """Number of effective dots in system."""
        n = 0
        for dot in self.dots.values():
            n += 1 if dot['isSC'] else dot['degeneracy']
        return n

    @property
    def nleads(self):
        """Number of leads in system."""
        return len(self.leads)

    @property
    def ncouplings(self):
        """Number of tunnel couplings (not incl. those to leads)."""
        return len(self.couplings)

    def add_dot(self, Ec, name=None, degeneracy=1, orbitals=0, isSC=False):    
        """Add a quantum dot to the system.

        Keyword Arguments:
        name (str): Key name (in self.dots) of the dot
        Ec (float): Charging energy of the dot.
        degeneracies (int): Orbital or spin degeneracy for each charge level.
        orbitals (float or float list): Orbital addition energy or SC
            gap. If length is greater than one, each entry is the orbital energy for
            each successive orbital. After all orbitals are filled, subsequent orbital
            energies are assumed equal to the last orbital energy. 
        isSC (bool): Whether or not dot is superconducting, in which case:
            --> degeneracy = degeneracy of odd parity quasiparticle state of dot
            --> orbitals = energy of odd parity lowest energy state / gap size

        Default settings:
            Dot with no degeneracy or confinement energy is added.
        """
        if isSC and not np.isscalar(orbitals):
            raise Exception('orbEnergies must be a scalar for superconducting islands,'
                + ' as it corresponds to odd parity lowest energy level.')
        if name is None: name = 'dot' + str(self.ndots)
        if any([name == n for n in {**self.dots, **self.leads}]):
            raise Exception("Dot or lead with name '" + name + "' is already defined.")
        self.dots[name] = {
            'Ec': Ec,
            'degeneracy': degeneracy,
            'orbitals': orbitals,
            'numCouplings': 0,
            'couplings': {},
            'isSC': isSC,
            'name': name, # For reverse searching in internal algorithms
            }
        # Assign an index range to dot so that it can be ordered for later calculations
        if self.dots[name]['isSC']:
            self._indices.append(name)
        else:
            self._indices.extend([name]*self.dots[name]['degeneracy'])
        if self.verbose:
            print("Dot added with name: " + str(name) + ".")
    
    def delete(self, name):
        """Delete dot/lead/coupling with 'name' and adjust _indices accordingly."""

        def del_coupling(name):
            """Deletes coupling and removes its corresponding information from self._indices."""
            ds = list(self.couplings[name]['dots'])
            # Delete each dot's entry in other dot's coupling list
            del self.dots[ds[0]]['couplings'][ds[1]]
            del self.dots[ds[1]]['couplings'][ds[0]]
            # Delete coupling itself
            del self.couplings[name]

        if name in self.couplings:
            del_coupling(name)
            return
        elif name in self.dots:
            del self.dots[name]
        elif name in self.leads:
            del self.leads[name]
        else:
            raise Exception('Dot/lead/coupling with name ' + name + 'does not exist!')
        for n,c in self.couplings:
            if name in c['dots']:
                warnings.warn('Coupling ' + n + 'involves dot/lead ' + name + ' and will be deleted.')
                del_coupling(c)
        self._indices.remove(name) # Adjust dot ordering/indexing if dot/lead is being removed

    def add_coupling(self, Em, t, dotnames, name=None):
        """Add tunnel coupling or mutual capacitance between two dots.
        
        Creates an entry in self.couplings containing a set of dots involved in the
        coupling in the first index, and a dictionary of properties of the coupling
        in the second index.

        Parameters:
        Em (float): Mutual capacitance energy (in ueV) between the two dots.
            Em = (e^2/Cm) / (C1*C2/Cm - 1)
        t (float): Tunnel coupling between the two dots (in ueV).
        dotnames (list of str): List of names of dots to be coupled.

        Keyword Arguments:
        name (str): Optional name associated with coupling, stored in dict. with Em/t
        """
        dns = list(dotnames)
        if any([name == n for n in self.couplings]):
            raise Exception("Coupling with name '" + name + "' is already defined.")
        if len(dns) > 2:
            raise Exception('Couplings can only be defined between two dots at a time!')
        for dot in dns:
            if dot not in self.dots:
                raise Exception(dot + ' is not a defined dot!')
        for c in self.couplings:
            if all([dn in c['dots'] for dn in dns]):
                del self.couplings[c]
                warnings.warn('Coupling already exists between ' + str(dns[0]) + ' and ' + str(dns[1])
                    + "! Overwriting with new coupling parameters.")
        for i,dot in enumerate(dns):
            self.dots[dot]['couplings'].update(
                {
                    dns[(i+1)%2]: {'Em': Em, 't': t} #Entry with OTHER dot's name as key
                }
            )
        self.couplings[name] = {
            'dots': set(dotnames),
            'Em': Em,
            't': t,
            'name': name # For reverse searching from dict. value
        }
        if self.verbose:
            print("Tunnel coupling and capacitance '" + name + "' added between dots: '" + str(dotnames) + '.')

    def add_lead(self, dots, t, name=None, level=0):
        """Add a lead to the system.

        Models lead as a quantum dot with 0 charging energy and many
        electrons, plus a tunnel coupling to certain dots.

        Parameters:
        dots (list): Dots (given by string keys) to which lead couples
        t (list): Tunnel couplings (t's) to dots, in same order as dots list
        
        Keyword Arguments:
        name (str): Key to access entry for this lead in self.leads
        level (float): Chemical potential (in ueV) of electrons in this lead

        Defaults:
        Creates lead with chemical potential zero.
        """
        if name == None:
            name = 'lead' + str(self.nleads)
        if any([name == n for n in {**self.dots, **self.leads}]):
            raise Exception("Lead or dot with name '" + name + "' is already defined.")
        self.leads[name] = {
            'couplings': {},
            'level': level,
            'isSC': False # Must be included for organization of charge states later.
        }
        # Create couplings dict. so 'leads' may be searched like dots
        for i,dot in enumerate(dots):
            self.leads[name]['couplings'][dot] = {
                'Em': 0,
                't': t[i]
            }
        self._indices.append(name)
        if self.verbose:
            print("Lead with chemical potential " + str(level) + "added which couples to dots: " + str(dots))

    def sc_state_ravel(self, dot, state):
        """Converts flattened index of SC island state to charge/orbital state.

        ## PG. 92-94 of MSc Logbook details derivation of this expression ##
        Given the flattened index of an SC island state, the charge and the
        orbital state of the dot are returned.

        Parameters:
        dot (str): keyword of SC dot in self.dots whose state is being considered.
        state (int): Position of dot state along its kwant lattice axis.

        Returns:
        charge (int): charge occupation of dot.
        orb (int): index of orbital state of dot. 
        """
        if dot not in self.dots:
            raise Exception('Input dot does not exist or is not a dot!')
        if not self.dots[dot]['isSC']:
            raise Exception('Input dot must be superconducting (isSC == True)!')
        d = [1,self.dots[dot]['degeneracy']]
        # Statement is written for more general d[0] in case other parity dependent
        # degeneracies may be implemented later
        charge = 2*(state // (d[0]+d[1])) + min([(state % (d[0]+d[1])) // d[0], 1])
        orb = state % (d[0]+d[1]) - (charge % 2)*d[0]
        return charge, orb

    def sc_state_unravel(self, dot, charge, orb = 0):
        """Convert charge/orbital state to unraveled index.

        ## PG. 92-94 of MSc Logbook details derivation of this expression ##
        Given the charge and orbital state of a dot, returns the index
        of that state in the corresponding kwant lattice.

        Parameters:
        dot (str): keyword of dot in self.dots whose state is being considered.
        charge (int): charge on SC island under consideration.
        orb (int): Index of occupied orbital in superconductor. If charge is even,
            orb must == 0 unless there is even parity degeneracy.

        Returns:
        int: state index of current charge/orbital state of SC island.
        """
        if dot not in self.dots:
            raise Exception('Input dot does not exist or is not a dot!')
        if not self.dots[dot]['isSC']:
            raise Exception('Input dot must be superconducting (isSC == True)!')
        # [even, odd] parity degeneracy of SC island
        d = [1,self.dots[dot]['degeneracy']]
        if orb > (1-charge%2)*d[0] + (charge%2)*d[1] - 1:
            raise Exception('Orbital index exceeds number of degenerate states possible!')
        return (charge//2)*(d[0] + d[1]) + (charge%2)*d[0] + orb 

    def dotstates(self,nameOrIndex,N):
        """List of charge states of dot/lead returned as [nameOrIndex,charge,isSC*orbital].

        Returns all possible states of dot with maximum charge N and minimum charge 0
        as a list of [nameOrIndex,charge,isSC*orbital] format states. Leads may contain
        as many as .ndots * N electrons.

        Parameters:
        nameOrIndex (int or str): Either index of dot/lead in ._indices, or keyword name
            in .dots or .leads
        N (int): Maximum number of charges to be allowed on dot

        Returns:
        list: List of all possible charge/orbital states of dot/lead in where each state is
            given as a list like [nameOrIndex,charge,isSC*orbital]
        """
        if np.isscalar(nameOrIndex):
            # Name of dot/lead
            name = self._indices[nameOrIndex]
            label = self._indices.index(name)
        elif isinstance(nameOrIndex, str):
            name = nameOrIndex
            label = name
        else:
            raise Exception('Input must be name keyword of dot/lead, or its index in ._indices.')
        if name in self.dots:
            dot = self.dots[name]
            if dot['isSC']:
                # Degeneracy of quasiparticle states on SC island
                deg = dot['degeneracy']
                # Number of possible SC island states
                numStates = (N//2)*(1+deg) + (N%2)*deg + 1
                # List comprehension generating all possible [charge,j,orbital] states
                return [
                    [
                        # Island index in self._indices or dot/lead name
                        label,
                        # Island charge 
                        2*(i // (1 + deg)) + min([(i % (1 + deg)), 1]), 
                        # 0 if even charge, index of deg. quasiparticle state if odd charge
                        i % (1 + deg) - min([i % (1 + deg),1]) % 2
                    ] 
                    for i in range(numStates)
                ]
            else:
                # If (effectively) a dot, states are just [charge,j,isSC = False]
                return [[label,i,0] for i in range(N+1)]
        elif name in self.leads:
            # If a lead, states are just [charge,j,isSC = False]
            return [[label,i,0] for i in range(N*self.ndots + 1)]
        else:
            raise Exception(
                "Object with name: " + name + "is in self._indices" 
                " but is not a defined lead or dot!"
                )
        
    def system_states(self, N):
        """Generate all possible charge/orbital states for given total charge N.

        Parameters:
        N (int): If system has no leads:
            Number of charges in system.
            Otherwise:
            Maximum number of charges per dot.

        Yields:
            numpy.array: Contains state of each dot in same order as in
            self._indices.
        """

        def unfixed_states(N):
            """Generates all charge states without a fixed total charge, wherein each
            effective dot may contain up to N electrons, and each lead can contain
            ALL electrons from each dot."""                
            # Take Cartesian product of all possible dot states to obtain a superset of 
            # all possible states, since the totalCharge may not = N.
            for state in product(*[self.dotstates(j,N) for j in range(len(self._indices))]):
                yield np.array(state)

        def is_possible_state(state, N):
            """Checks if a charge state, given as a numpy array, is allowable for fixed total N."""
            # Sum state tuple to find charge
            totalCharge = state[:,1].sum()
            # State is not possible if total charge does not equal N (no leads) or N*self.ndots (leads)
            if (totalCharge != N and self.nleads == 0) or (totalCharge != N*self.ndots and self.nleads != 0):
                return False
            # Find index of 1st occurrence of all degeneracy dots, even those across different 'real' dots
            degIndices = {self._indices.index(n) for n in self._indices if self._indices.count(n) > 1}
            # Don't bother checking differences between degenerate dot states if there are no degeneracies
            if len(degIndices) == 0:
                return True
            # Sort degeneracy dot charges by which real dot they belong to
            dotCharges = [state[state[:,0] == i,1] for i in degIndices]
            return not any([any([abs(d[0]-d[1]) > 1 for d in combinations(dc,2)]) for dc in dotCharges])

        for state in unfixed_states(N):
            if is_possible_state(state, N):
                yield state
    

    ### WILL NEED DEBUGGING ###
    def dimension(self, N):
        """Return dimension (int) of Hilbert space for a given number of charges N in the system."""
        if self.verbose: ti = time.clock()
        dim = len(list(self.system_states(N)))
        if self.verbose:
            print('Time to generate states was: ' + str(time.clock() - ti) + 's.')
        return dim

    def get_hamiltonian(self, N, gates):
        """Retrieve system Hamiltonian for given maximum charge and current dots/leads.

        Returns a list (matrix) whose first entry in each element is the Hamiltonian matrix
        element for the dot system with total charge N (or maximum charge N per dot if leads
        are present) and whose second entry is the involved state(s) given by system_states(N).

        Parameters:
        N (int): Total charge of all dots in system, or the maximum charge per dot if leads are 
            present.

        Keyword Arguments:
        gates: Dictionary containing dot names as keywords and reduced gate voltages as values.

        Returns:
        ham (np.array): Hamiltonian matrix of the system.
        states: List of dictionaries of the charge configuration corresponding to each state.
            Order of list matches the diagonal of ham.
        """

        if self.nleads > 0:
            warnings.warn("At least one dot has leads, so N cannot be fixed," 
                          " and will be interpreted as the maximum charge per dot.")

        dim = self.dimension(N)
        ham_diag = []  #Initialize Hamiltonian diagonal
        def onsite(state, gates):
            """Calculates onsite energy for a given charge configuration of the DotSystem."""

            def objindices(name):
                """Returns list of indices in state corresponding to dot with name."""
                return [i for i,n in enumerate(state[:,1]) if n == name]

            ns = {} # Initialize dictionary of charge on each dot
            stateEnergy = 0 # Initialize energy of state to be returned
            for dot, name in self.dots.items():
                indices = objindices(name)
                charges = np.array([state[i][1] for i in indices])
                if dot['isSC']:    
                    stateEnergy += (charges[0]%2)*dot['orbitals']
                elif np.isscalar(dot['orbitals']):
                    # If addition energy is the same for every orbital, we need only multiply
                    # the total dot charge by this orbital energy to calculate total
                    # non-charging-related energy
                    stateEnergy += sum(charges)*dot['orbitals']
                else:
                    # For multiple orbital energies, 
                    numFullOrbitals = charges//dot['degeneracy']
                    orbs = np.array(dot['orbitals'])
                    # If all orbitals specified in 'orbitals' are full, then higher orbital energy
                    # are assumed equal to the highest specified orbital energy
                    if numFullOrbitals > len(orbs):
                        np.append(orbs, [orbs[-1]]*(numFullOrbitals - len(orbs)))
                    # Add all orbital energies for each degeneracy quantum number, then sum the result
                    stateEnergy += sum([sum(orbs[0:subcharge]) for subcharge in charges])
                # Add charging energy of dot
                ns.update({name: sum(charges)-gates[name]})
                stateEnergy += dot['Ec']*ns[name]**2
            # Add mutual capacitance energy between dots
            for c in self.couplings.values():
                stateEnergy += c['Em'] * ns[c['dots'][0]] * ns[c['dots'][1]]
            # Add chemical potential energy of leads
            for lead, name in self.leads.items():
                stateEnergy += state[objindices(name)][0]*lead['level'] 
            return stateEnergy

        ham_diag = [[onsite(state,gates),state] for state in self.system_states(N)]

        # def is_neighbour(state1,state2):
        #     "Returns whether or not state1 & state2 have a non-zero matrix element (bool)."
        #     diffDots = {} # Set containing names of dots whose charge differs between states
        #     for s1, s2 in zip(state1, state2):
        #         # Set eff. dot label to 'SC' if SC, else leave as degeneracy index
        #         # for normal dots, or as 'lead' for leads.
        #         i1 = 'SC' if s1[0] in self.dots and self.dots[s1[0]]['isSC'] else s1[2]
        #         i2 = 'SC' if s2[0] in self.dots and self.dots[s2[0]]['isSC'] else s2[2]
        #         # Check if dot/lead names are equal and if degeneracy index is equal
        #         if s1[0] == s2[0] and i1 == i2:
        #             diff = abs(s1[1]-s2[1]) # Difference in dot charge between states
        #             if diff > 1:
        #                 # Charge on dot differs by >=2 between states
        #                 return False
        #             elif diff = 1:
        #                 1 == 1
        #                 # Charge has been tranferred between these 2 dots

        #                 diffDots.update({s1[0],s2[0]})
        #         if len(diffDots) > 2:
        #             return False
        #     if len(diffDots) == 2 and any([c['names'] == diffDots for c in self.couplings]):
        #         # Only two (effective) dots differ in charge, and they have a coupling with each other
        #         return True
        #     return False


        
    # def get_num_op(self, dotname, N):
    #     """Return number operator for a given dot.

    #     Parameters:
    #     dotname (str): key for dot in self.dots.
    #     N (int): Total charge in system if there are no leads,
    #         or maximum charge per dot if there are leads.
    #         Required because system's dimensionality is dependent on N.
        
    #     Returns:
    #     ndarray: Matrix form of the number operator for desired dot.
    #     """
    #     self.indexer()
    #     dim = self.dimension(N)
    #     dot = self.dots[dotname]

    #     numOp = np.zeros((dim,dim))
    #     for i in range(dim):
    #         pos = list(self.sys.sites[i].pos)
    #         if self.dots[dotname]['type'] == 'superconducting':
    #             numOp[i,i] = self.sc_state_mapper(dot, pos[dot['indices'][0]], want='charge')
    #         else:
    #             numOp[i,i] = sum(pos[dot['indices'][0]: dot['indices'][1] + 1])
    #     return numOp

    # Built-in analysis functions here?


"""
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
FOR TESTING
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
"""

def main():
    x = 0
    system = DotSystem(verbose=True)
    #system.add_dot(100,degeneracy=1,orbitals=100,isSC = False)
    # system.add_dot(20,degeneracy=2,orbitals=0,isSC=False)
    # system.add_dot(30,degeneracy=4,orbitals=[3,5,6],isSC=False)
    system.add_dot(400,name='SCdot',degeneracy=5,orbitals=100,isSC=True)
    system.add_dot(50,name='SCdot2',degeneracy=4,orbitals=200,isSC=True)
    
    print(system.ndotseff)
    print(system.ndots)
    N = 4
    npoints = 101
    print('System dimension is: ' + str(system.dimension(N)))

    # if 1 == 0:    
    #     system.to_kwant(N = N)
    #     gatespace = {}
    #     for dot in system.dots: gatespace[dot] = np.linspace(0,N,npoints)
    #     spacing = gatespace['dot0'][1] - gatespace['dot0'][0]
    #     numOp0 = system.get_num_op('dot0', N)
    #     numOp1 = system.get_num_op('dot1', N)
    #     GSEnergy = np.zeros((npoints,npoints))
    #     number   = np.zeros((2, npoints, npoints))

    #     print('About to start diagonalizing a bunch of Hamiltonians...')
    #     for i, g1 in enumerate(gatespace['dot0']):
    #         for j, g2 in enumerate(gatespace['dot1']):
    #             gates = {}
    #             gates['dot0'] = g1
    #             gates['dot1'] = g2
    #             params = (gates, )

    #             if i == 0 and j == 0: ti = time.clock()
    #             H = system.sys.hamiltonian_submatrix(params)
    #             # try:
    #             #     E, vecs = sp.linalg.eigsh(H, k=1, which = 'SM')
    #             #     number[0,i,j] = np.dot(vecs.conj().T, np.dot(numOp0, vecs))[0][0]
    #             #     number[1,i,j] = np.dot(vecs.conj().T, np.dot(numOp1, vecs))[0][0]
    #             # except:
    #             # warnings.warn("Sparse diagonalization did not converge for " + str(g1) + ", " + str(g2) + ". Attempting exact   diagonalization.")
    #             Etot, vecstot = np.linalg.eigh(H)
    #             E = Etot[0]
    #             vecs = vecstot[:,0]
    #             number[0,i,j] = np.dot(vecs.conj().T, np.dot(numOp0, vecs))
    #             number[1,i,j] = np.dot(vecs.conj().T, np.dot(numOp1, vecs))

    #             if i == 0 and j == 2:
    #                 t = (time.clock() - ti)/3
    #                 print("Time to generate and diagonalize one Hamiltonian is: " + str(t) + "\n"
    #                     "(--> time for completion is: " + str(int((npoints * npoints - 1)*t//60)) + "m" + str(int((npoints * npoints - 1)*t%60)) + "s)")
    #             GSEnergy[i,j] = E
    #             if j % (npoints + 1) == 0: print('done up to: ' + str(i) + ' rows.\r')
        
    #     print('Done diagonalizing Hamiltonians.')

    #     # Quantum capacitance
    #     QC0 = np.zeros((npoints-1,npoints))
    #     QC1 = np.zeros((npoints, npoints-1))
    #     for j in range(npoints):
    #         QC0[:,j] = np.diff(number[0,j,:].T)/spacing
    #         QC1[j,:] = np.diff(number[1,:,j].T)/spacing


    # print('System dimension is: ' + str(system2.dimension(N)))

    # if 1 == 1:    
    #     system2 = DotSystem(print = True)
    #     system2.add_dot(levels = [0], delta = 100, Ec = 150, name = 'dot0')
    #     system2.add_dot(Ec = 200, name = 'dot1')
    #     system2.add_lead(['dot0'], [10], name = 'lead0', level = 0)
    #     system2.add_lead(['dot1'], [10], name = 'lead1', level = 0)
    #     system2.add_coupling(Em = 50, t = 10, dot1name = 'dot0', dot2name = 'dot1')
    #     system2.to_kwant(N = N)
    #     numOp02 = system2.get_num_op('dot0', N)
    #     numOp12 = system2.get_num_op('dot1', N)
    #     GSEnergy2 = np.zeros((npoints,npoints))
    #     number2   = np.zeros((2, npoints, npoints))

    #     print('About to start diagonalizing a bunch of Hamiltonians...')
    #     for i, g1 in enumerate(gatespace['dot0']):
    #         for j, g2 in enumerate(gatespace['dot1']):
    #             gates = {}
    #             gates['dot0'] = g1
    #             gates['dot1'] = g2
    #             params = (gates,)

    #             if i == 0 and j == 0: ti = time.clock()
    #             H = system2.sys.hamiltonian_submatrix(params)
    #             # try:
    #             #     E, vecs = sp.linalg.eigsh(H, k=1, which = 'SM')
    #             #     number[0,i,j] = np.dot(vecs.conj().T, np.dot(numOp0, vecs))[0][0]
    #             #     number[1,i,j] = np.dot(vecs.conj().T, np.dot(numOp1, vecs))[0][0]
    #             # except:
    #             # warnings.warn("Sparse diagonalization did not converge for " + str(g1) + ", " + str(g2) + ". Attempting exact   diagonalization.")
    #             Etot2, vecstot2 = np.linalg.eigh(H)
    #             E2 = Etot2[0]
    #             vecs2 = vecstot2[:,0]
    #             number2[0,i,j] = np.dot(vecs2.conj().T, np.dot(numOp02, vecs2))
    #             number2[1,i,j] = np.dot(vecs2.conj().T, np.dot(numOp12, vecs2))

    #             if i == 0 and j == 2:
    #                 t = (time.clock() - ti)/3
    #                 print("Time to generate and diagonalize one Hamiltonian is: " + str(t) + "\n"
    #                     "(--> time for completion is: " + str(int((npoints * npoints - 1)*t//60)) + "m" + str(int((npoints * npoints - 1)*t%60)) + "s)")
    #             GSEnergy2[i,j] = E
    #             if j % (npoints + 1) == 0: print('done up to: ' + str(i) + ' rows.\r')
        
    #     print('Done diagonalizing Hamiltonians.')

    #     # Quantum capacitance
    #     QC02 = np.zeros((npoints-1,npoints))
    #     QC12 = np.zeros((npoints, npoints-1))
    #     for j in range(npoints):
    #         QC02[:,j] = np.diff(number2[0,j,:].T)/spacing
    #         QC12[j,:] = np.diff(number2[1,:,j].T)/spacing

    #     fig, axs = plt.subplots(2,2, sharex = True, sharey = True)

    #     cmap = plt.get_cmap('viridis')
    #     # fig.subplots_adjust(wspace = 0.3) # Increase spacing between plots
    #     # fig.set_size_inches(10,10) # Increase size of output image (WxL)

    #     ax = axs[0,0]
    #     a = ax.pcolormesh(gatespace['dot0'], gatespace['dot1'], abs(GSEnergy.T - GSEnergy2.T), cmap = cmap)
    #     fig.colorbar(a, ax = ax)
    #     ax.set_xlabel('n_g1')
    #     ax.set_ylabel('n_g2')
    #     ax.set_title('GS Energy (ueV)')

    #     ax = axs[0,1]
    #     a = ax.pcolormesh(gatespace['dot0'], gatespace['dot1'], abs(((QC0[:,0:-1] + QC1[0:-1,:])/2) - ((QC02[:,0:-1] + QC12[0:-1,:])/2)), cmap = cmap)
    #     fig.colorbar(a, ax = ax)
    #     ax.set_xlabel('n_g1')
    #     ax.set_ylabel('n_g2')
    #     ax.set_title('Sum of Qc\'s')

    #     ax = axs[1,0]
    #     b = ax.pcolormesh(gatespace['dot0'], gatespace['dot1'], abs(number[0,:,:].T - number2[0,:,:].T), cmap = cmap)
    #     fig.colorbar(b, ax = ax)
    #     ax.set_xlabel('n_g1')
    #     ax.set_ylabel('n_g2')
    #     ax.set_title('d<n1>/dn_g1')

    #     ax = axs[1,1]
    #     c = ax.pcolormesh(gatespace['dot0'], gatespace['dot1'], abs(number[1,:,:].T - number2[1,:,:]), cmap = cmap)
    #     fig.colorbar(c, ax = ax)
    #     ax.set_xlabel('n_g1')
    #     ax.set_ylabel('n_g2')
    #     ax.set_title('d<n2>/dn_g2')

    #     plt.show()

    #     print(system.couplings)
    
    
if __name__ == '__main__':
    main()