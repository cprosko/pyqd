"""
qdsystem:
Package for simulating quantum dot systems with or without leads.

Work flow of a qdsystem calculation:
    1. Initialize an instance of the qdsystem class.
    2. Add quantum dots with their corresponding charging energies and other properties with self.add_dot()
    3. Add tunnel couplings between dots with self.add_coupling().
    4. Add leads to dots of your choice with self.add_lead().
    5. Generate the system states for a given maximum charge (per dot if there are leads) with self.get_states(N)
    6. Calculate properties of the system using other functions.
"""

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
    ndotssc: number of superconducting type dots
    """
    def __init__(self, verbose = True):
        """
        Constructor for DotSystem class.

        Parameters:
        verbose (bool): whether or not to print results of calling methods
        """
        # Dictionary of dots and leads and their parameters and couplings
        self.objects = {}   
        # Dictionary of all possible system states, searchable by total charge N    
        self.states = {}
        # Dictionary containing properties from ._sysTemp in numpy array format
        self._sys = {}   
        # List containing system information in mutable list format 
        # Formatted as ._sysTemp[i] = ['name',Ec,isSC,isLead,orbitals,ts,Ems]    
        self._sysTemp = []     
        # Suppresses printing function results if False   
        self.verbose = verbose
        # Dictionary of bools: whether system has been finalized in current 
        # configuration for a given charge (key).
        self.isFinalized = {}
    
    def __str__(self):
        """Print class type and attributes when qdsystem is called with print()."""
        output = "qdsystem class object with: \n"
        output += str(self.ndots) + " total dots, with "
        output += str(self.nleads) + " leads."
        return output

    @property
    def ndots(self):
        """Number of dots in system."""
        return sum([not v['isLead'] for v in self.objects.values()])

    @property
    def ndotseff(self):
        """Number of effective dots in system."""
        return len(self._sysTemp)

    @property
    def ndotssc(self):
        """Number of superconducting islands in system."""
        return sum([v[2] for v in self._sysTemp])

    @property
    def nleads(self):
        """Number of leads in system."""
        return sum([v[3] for v in self._sysTemp])

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
            raise Exception(
                'orbEnergies must be a scalar for superconducting islands,'
                + ' as it corresponds to odd parity lowest energy level.')
        if name is None: name = 'dot' + str(self.ndots)
        if any([name == n for n in self.objects]):
            raise Exception("Dot or lead with name '" + name + "' is already defined.")
        self.objects[name] = {
            'Ec': Ec,
            'degeneracy': degeneracy,
            'orbitals': orbitals,
            'numCouplings': 0,
            'couplings': {},
            'isSC': isSC,
            'isLead': False,
            'name': name, # For reverse searching in internal algorithms
            }
        # Add dot to collection of system objects, accounting for degeneracy
        numEffDots = isSC + (1-isSC)*degeneracy
        orbs = [orbitals]*degeneracy if isSC else orbitals
        self._sysTemp.extend([[name,Ec,isSC,False,orbs,{},{}]]*numEffDots)
        if self.verbose:
            print("Dot added with name: " + str(name) + ".")

    def add_coupling(self, Em, t, dotnames):
        """Add tunnel coupling or mutual capacitance between two dots.
        
        Creates an entry in self.couplings containing a set of dots involved in the
        coupling in the first index, and a dictionary of properties of the coupling
        in the second index.

        Parameters:
        Em (float): Mutual capacitance energy (in ueV) between the two dots.
            Em = (e^2/Cm) / (C1*C2/Cm - 1)
        t (float): Tunnel coupling between the two dots (in ueV).
        dotnames (tuple or set of strings): List of names of dots to be coupled.
        """
        dns = list(dotnames)
        if any([obj not in self.objects for obj in dns]):
            raise Exception(dot + ' is not a defined dot!')
        for d1,d2 in combinations(*dns,2):
            if d2 in self.objects[d1]['couplings']:
                warnings.warn(
                    'Coupling already exists between objects ' + d1 + ' and ' + d2
                    + "! Overwriting with new coupling parameters."
                    )
            else:
                self.objects[d1]['numCouplings'] += 1
                self.objects[d2]['numCouplings'] += 1
            # Add entry to 'couplings' with other dot's name as key
            c = {'Em': Em, 't': t}
            self.objects[d1]['couplings'].update({d2: c})
            self.objects[d2]['couplings'].update({d1: c})
            # Update self._sysTemp with coupling information
            i1 = [i for i,v in self._sysTemp if v[0] == d1]
            i2 = [i for i,v in self._sysTemp if v[0] == d2]
            for i,j in zip(i1,i2):
                self._sysTemp[i][5].update({j: t})
                self._sysTemp[i][6].update({j: Em})
                self._sysTemp[j][5].update({i: t})
                self._sysTemp[j][6].update({i: Em})

        if self.verbose:
            print(
                "Tunnel coupling " + str(t) + " and mutual capacitance "
                + str(Em) + "added between dots: '" + str(dotnames) + '.'
                )

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
        if any([name == n for n in self.objects]):
            raise Exception("Lead or dot with name '" + name + "' is already defined.")
        index = self.ndots + self.nleads
        self.objects[name] = {
            'couplings': {},
            'level': level,
            'isSC': False, # Must be included for organization of charge states later.
            'isLead': True,
            'name': name,
        }
        self._sysTemp.append([name,0,False,True,level,1,{},{}]) # Add lead to internal list of system objects
        # Create couplings dict. so 'leads' may be searched like dots
        for i,dot in enumerate(dots):
            self.add_coupling(0,t[i],(name,dot))
        if self.verbose:
            print("Lead with chemical potential " + str(level) + "added which couples to dots: " + str(dots))

    def delete(self, *names):
        """Delete dot/lead/coupling with [name].

        Adjusts ._sysTemp accordingly for deleted dot/lead.

        Parameters:
        names (string or tuple of strings): if string, deletes
            dot or lead with same name, if tuple of strings, deletes
            all couplings between named objects.
        """
        def del_coupling(*names):
            """Deletes coupling between objects in names."""
            if len(names) == 1:
                name = names[0]
                for n in [n for n in self.objects if name in self.objects[n]['couplings']]:
                    del self.objects[n]['couplings'][name]
            else:
                for n1,n2 in permutations(*names,2):
                    del self.objects[n1]['couplings'][n2]

        if len(names) == 1:
            n = names[0]
            del self.objects[n]
            del_coupling(n)
            for i in [i for i,v in enumerate(self._sysTemp) if v[0] == n]:
                del self.sysTemp[i]
            # Dictionary of system states is no longer valid after object is removed
            self.states = {}
        else:
            del_coupling(*names)

    def finalize(self):
        """Port system information from ._sysTemp to dictionary of numpy arrays in ._sys"""
        l = len(self.objects) # Number of eff. dots / leads in system
        # Convert ._sysTemp to numpy array of all objects
        sys = np.array(self._sysTemp,dtype=object) 
        # First translate each object's properties into numpy array
        self._sys.update({
            'name': np.array(sys[:,0],dtype=str),
            'Ec': np.array(sys[:,1],dtype=float),
            'isSC': np.array(sys[:,2],dtype=bool),
            'isLead': np.array(sys[:,3],dtype=bool),
            'orbs': np.array(sys[:,4],dtype=list),
            'Em': np.zeros((l,l)),
            't': np.zeros((l,l)),
        })
        # Next, translate couplings into numpy array in ._sys as well
        for i,j in combinations(range(l),2):
            print(i)
            print(j)
            t = self._sysTemp[i][5]
            Em = self._sysTemp[i][6]
            self._sys['t'][i,j] = t[j] if j in t else 0
            self._sys['Em'][i,j] = Em[j] if j in Em else 0
        # Finally, symmetrize the 't' and 'Em' arrays:
        self._sys['t'] = symmetrize(self._sys['t'])
        self._sys['Em'] = symmetrize(self._sys['Em'])
        
    def get_states(self, N):
        """Generate all possible charge/orbital states for given total charge N.

        Parameters:
        N (int): If system has no leads:
            Number of charges in system.
            Otherwise:
            Maximum number of charges per dot.

        Returns:
        numpy.array: Contains state of each (quasi-)dot/lead in same order as in
            self._sys.
        """
        self.finalize()
        sys = self._sys 
        nObjs = self.ndotseff
        nMax = N*self.ndots if self.nleads > 0 else N
        # List of all indices of 'quasi'-dots comprising degenerate dots
        degIndices = [
            [i for i,v in enumerate(sys['name']) if sys['name'][i] == n]
            for n in sys['name'] if (sys['name'] == n).sum() > 1
        ]

        def isEcEnforced(chgs):
            """Ensures degenerate dots don't have extra charge in degenerate orbital."""
            return any([
                any([abs(d[0]-d[1]) > 1 for d in combinations(chgs[inds],2)])
                for inds in degIndices
                ])

        def partitions(n,k,l=0,nc=None):
            if nc == None:
                nc = n
            if k < 1:
                # If no integers are left, we are done.
                return
            if k == 1:
                if nc >= l:
                    yield (nc,)
                # If only one integer is left, and the remainder is larger than
                # the minimum size l, we are done.
                return
            for i in range(l,n+1):
                # Call partitions with l=i to ensure that no tuple is yielded twice,
                # as this enforces i being the largest integer
                for r in partitions(n,k-1,l=i,nc=n-i):
                    s = (i,) + tuple(r)
                    # If len(s) == n, we have an (undordered) charge state
                    print(s)
                    if len(s) == n:
                        # Find all ordered charge states from unordered states
                        for p in enumerate(multiset_permutations(s)):
                            # If degenerate dots exist, enforce charging energy.
                            if self.ndotseff != self.ndots and not isEcEnforced(p):
                                continue
                            yield p
                        continue
                    yield s
        
        def dotstate(i,charge):
            """Returns all possible dot/lead (of index i) states for given charge."""
            deg = len(sys['orbs'][i]) if charge%2 and sys['isSC'][i] else 1
            return [[charge,j] for j in range(deg)]

        ti = time.clock()
        states = [chgs for chgs in partitions(nMax,nObjs)]
        print(states)
        print(len(states))
        print(str(time.clock() - ti) + 's')


    
    def dimension(self, N):
        """Return dimension (int) of Hilbert space for a given number of charges N in the system."""
        # Generate system states for N total charge if it has not already been calculated.
        if N not in self.states:
            if self.verbose:
                ti = time.clock()
            self.states[N] = np.array(list(self.get_states(N)))
            if self.verbose:
                print('Time to generate states was: ' + str(time.clock() - ti) + 's.')
        return len(self.states[N])

        


    # def get_hamiltonian(self, N, gates, storeStates = True):
    #     """Retrieve the system Hamiltonian for given total charge N and gate voltages."""
    #     if self.nleads > 0 and self.verbose:
    #         warnings.warn(
    #             "At least one dot has leads, so N cannot be fixed," 
    #             " and will be interpreted as the maximum charge per dot."
    #             )
    #     if N not in self.states:
    #         self.states[N] = np.array(list(self.system_states(N)))
    #     states = self.states[N]
    #     indices = np.array(self._indices)
    #     dots = self.dots
    #     leads = self.leads

    #     # charging energy first:
    #     # in each state, pick out only dots
    #     dotcharges = states[:,np.isin(indices,self.dots),1]
    #     # Find corresponding charging energies:

        # def onsite(state, gates):
        #     """Calculates onsite energy for a given charge configuration of the DotSystem."""

        #     def objindices(name):
        #         """Returns list of indices in state corresponding to dot with name."""
        #         return [i for i,n in enumerate(state[:,1]) if n == name]

        #     ns = {} # Initialize dictionary of charge on each dot
        #     stateEnergy = 0 # Initialize energy of state to be returned
        #     for dot, name in self.dots.items():
        #         indices = objindices(name)
        #         charges = np.array([state[i][1] for i in indices])
        #         if dot['isSC']:    
        #             stateEnergy += (charges[0]%2)*dot['orbitals']
        #         elif np.isscalar(dot['orbitals']):
        #             # If addition energy is the same for every orbital, we need only multiply
        #             # the total dot charge by this orbital energy to calculate total
        #             # non-charging-related energy
        #             stateEnergy += sum(charges)*dot['orbitals']
        #         else:
        #             # For multiple orbital energies, 
        #             numFullOrbitals = charges//dot['degeneracy']
        #             orbs = np.array(dot['orbitals'])
        #             # If all orbitals specified in 'orbitals' are full, then higher orbital energy
        #             # are assumed equal to the highest specified orbital energy
        #             if numFullOrbitals > len(orbs):
        #                 np.append(orbs, [orbs[-1]]*(numFullOrbitals - len(orbs)))
        #             # Add all orbital energies for each degeneracy quantum number, then sum the result
        #             stateEnergy += sum([sum(orbs[0:subcharge]) for subcharge in charges])
        #         # Add charging energy of dot
        #         ns.update({name: sum(charges)-gates[name]})
        #         stateEnergy += dot['Ec']*ns[name]**2
        #     # Add mutual capacitance energy between dots
        #     for c in self.couplings.values():
        #         stateEnergy += c['Em'] * ns[c['dots'][0]] * ns[c['dots'][1]]
        #     # Add chemical potential energy of leads
        #     for lead, name in self.leads.items():
        #         stateEnergy += state[objindices(name)][0]*lead['level'] 
        #     return stateEnergy

        # ham_diag = [[onsite(state,gates),state] for state in self.states[N]]

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

def symmetrize(mat):
    """Symmetrizes a numpy array like object."""
    return mat + mat.T - np.diag(mat.diagonal())

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
    system.add_dot(20,degeneracy=2,orbitals=0,isSC=False)
    system.add_dot(400,name='SCdot',degeneracy=10,orbitals=100,isSC=True)
    system.get_states(2)

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