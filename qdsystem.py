"""
qdsystem:
Package for simulating quantum dot systems with or without leads.

Work flow of a qdsystem calculation:
    1. Initialize an instance of the DotSystem class.
    2. Add quantum dots with their corresponding charging energies
       and other properties with self.add_dot()
    3. Add tunnel couplings between dots with self.add_coupling().
    4. Add leads to dots of your choice with self.add_lead().
    5. Generate the system states for a given maximum charge
       (per dot if there are leads) with self.get_states(N).
    6. Calculate properties of the system using other functions.
"""

### TO DO:
# 1. Optimize partitions() function
# 2. Vectorize calculation of 'diffs' in .areNeighbours(), currently uses list comp.
# 3. Finish adding 2e tunnel couplings

import numpy as np
import matplotlib.pyplot as plt
import warnings
import time

from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from itertools import product, combinations, permutations
from sympy.utilities.iterables import multiset_permutations

class DotSystem:
    """
    Class for simulating systems of multiple quantum dots.

    Attributes:
    objects: Dictionary of properties for the system's dots/SC islands/leads
    states: Sorted dictionary of arrays of all possible system states, indexed
        by the total charge (per dot if there are leads)
    leads: Sorted dictionary of leads and their properties
    verbose: whether or not to print results of calling methods
    ndots: number of dots in system
    nleads: number of leads in system
    ndotssc: number of superconducting type dots
    """
    def __init__(self, verbose=True):
        """
        Constructor for DotSystem class.

        Parameters:
        verbose (bool): whether or not to print results of calling methods
        """
        # Dictionary of dots and leads and their parameters and couplings
        self.objects = {}   
        # Dictionary of all possible system states, searchable by total charge N    
        self.states = {}
        # Dictionary of all tunneling Hamiltonians, with N as key
        self.__tunnelingHam = {}
        # Dictionary of all orbital + mutual capacitance diagonals, with N as key
        self.__diagElements = {}
        # Dictionary containing properties from .__sysTemp in numpy array format
        self.__sys = {}   
        # List containing system information in mutable list format 
        # Formatted as .__sysTemp[i] = ['name',Ec,isSC,isLead,orbitals,deg.,ts,Ems]    
        self.__sysTemp = []     
        # Suppresses printing function results if False   
        self.verbose = verbose
        # Whether system has been finalized into processable form.
        self.__isFinalized = False
    
    def __str__(self):
        """Print class type and attributes when qdsystem is called with print()."""
        output = (
            'qdsystem class object with: \n'
            +str(self.ndots)+' total dot{plr}, '.format(plr='s' if self.ndots > 1 else '')
            +str(self.ndotssc)+' of which are superconducting and '
            +str(self.ndots-self.ndotssc)+' of which are normal, as well as '
            +str(self.nleads)+' lead{plr}.'.format(plr='s' if self.nleads > 1 else '')
        )
        return output

    @property
    def ndots(self):
        """Number of dots in system."""
        return sum([not v['isLead'] for v in self.objects.values()])

    @property
    def __ndotseff(self):
        """Number of effective dots in system, including degeneracy on normal dots."""
        return len(self.__sysTemp)

    @property
    def ndotssc(self):
        """Number of superconducting islands in system."""
        return sum([v[2] for v in self.__sysTemp])

    @property
    def nleads(self):
        """Number of leads in system."""
        return sum([v[3] for v in self.__sysTemp])
        
    def __cast_as_sclr(self,n):
        """Converts length 1 arrays to scalars, and leaves scalars as is."""
        if np.isscalar(n):
            return n
        elif len(n) == 1:
            return n[0]
        else:
            raise Exception("Input must be an array-like of length 1 or a scalar!")

    def add_dot(self,Ec,
        name=None,degeneracy=1,orbitals=0,u=0,isSC=False,spin=False):    
        """Add a quantum dot to the system.

        Keyword Arguments:
        name (str): Key name (in self.dots) of the dot
        Ec (float): Charging energy of the dot.
        degeneracy (int): Orbital or spin degeneracy for each charge level.
            If spin==True, this is the degeneracy PER spin.
        orbitals (float or float list): Orbital addition energy or SC
            gap. If length is greater than one, each entry is the orbital energy for
            each successive orbital. After all orbitals are filled, subsequent orbital
            energies are assumed equal to the last orbital energy. 
        u (float): (Even,n)-->(Odd,n+1) parity tunneling multiplier (coherence factor u)
            The corresponding v = sqrt(1-u^2) is automatically inferred from this.
            If u==0, equal electron-like and hole-like tunneling is assumed.
        isSC (bool): Whether or not dot is superconducting, in which case:
            --> degeneracy = degeneracy of odd parity quasiparticle state of dot (/2 if spin)
            --> orbitals = energy of odd parity lowest energy state / gap size
        spin (bool): Whether or not quantum states on the dot are spinful
            (with spin-1/2 of course)
        """
        if isSC and not np.isscalar(orbitals):
            raise Exception(
                'orbEnergies must be a scalar for superconducting islands,'
                + ' as it corresponds to odd parity lowest energy level.')
        if name is None:
            name = 'dot' + str(self.ndots)
        if any([name == n for n in self.objects]):
            raise Exception("Dot or lead with name '" + name + "' is already defined.")
        if spin:
            deg = 2*degeneracy
            degType = 'spin'
        else:
            deg = degeneracy
            degType = 'none'
        self.objects[name] = {
            'Ec': Ec,
            'degeneracy': deg,
            'orbitals': orbitals,
            'numCouplings': 0,
            'couplings': {},
            'twoeCouplings': {},
            'u': u,
            'isSC': isSC,
            'isLead': False,
            'spin': degType,
            'name': name, # For reverse searching in internal algorithms
            }
        # Add dot to collection of system objects, accounting for degeneracy
        numEffDots = isSC + (1-isSC)*deg
        # Account for the fact that 0 electrons costs 0 orbital energy
        if not isSC:
            orbitals = [0,orbitals] if np.isscalar(orbitals) else [0] + orbitals
        self.__sysTemp.extend([[
            name,
            Ec,
            isSC,
            False, # Dots are not leads
            orbitals, # Odd parity free energy if isSC, else list of orbital energies
            degType, # Degree of freedom corresponding to degeneracy or 'none'
            deg, # Degeneracy of orbitals (quasiparticle states) for dots (SC islands).
            {}, # Dictionary of all dots with tunnel couplings
            {}, # Dictionary of all dots with mutual capacitances
            {}, # Dictionary of all dots with 2e- tunneling
            u, # Electron-like coherence factor or lack of coherence factors (if u == 0)
            ]]*numEffDots)
        if self.verbose:
            print("Dot added with name: " + str(name) + ".")

    def add_coupling(self, Em, t, dotnames,twoe=False):
        """Add tunnel coupling or mutual capacitance between two dots.
        
        Creates an entry in self.couplings containing a set of dots involved in the
        coupling in the first index, and a dictionary of properties of the coupling
        in the second index.

        Parameters:
        Em (float): Mutual capacitance energy (in ueV) between the two dots.
            Em = (e^2/Cm) / (C1*C2/Cm - 1)
        t (float): Tunnel coupling between the two dots (in ueV).
        dotnames (tuple or set of strings): List of names of dots to be coupled.

        Keyword Arguments:
        twoe (bool): Whether or not hopping is two electron or single electron.
        """
        dns = list(dotnames)
        if any([obj not in self.objects for obj in dns]):
            raise Exception(dns + ' contains undefined dots or leads!')
        for d1,d2 in combinations(dns,2):
            if (
                d2 in self.objects[d1]['couplings'] and not twoe
                or d2 in self.objects[d1]['twoeCouplings'] and twoe
            ):
                warnings.warn(
                    'Coupling already exists between objects ' + d1 + ' and ' + d2
                    + "! Overwriting with new coupling parameters."
                    )
            else:
                self.objects[d1]['numCouplings'] += 1
                self.objects[d2]['numCouplings'] += 1
            # Add entry to 'couplings' with other dot's name as key
            if twoe:
                c = {'Em': Em, 't': 0}
                self.objects[d1]['twoeCouplings'][d2] = t
                self.objects[d2]['twoeCouplings'][d1] = t
            else:
                c = {'Em': Em, 't': t}
            self.objects[d1]['couplings'].update({d2: c})
            self.objects[d2]['couplings'].update({d1: c})
            # Update self.__sysTemp with coupling information
            i1 = [i for i,v in enumerate(self.__sysTemp) if v[0] == d1]
            i2 = [i for i,v in enumerate(self.__sysTemp) if v[0] == d2]
            for i,j in zip(i1,i2):
                if twoe:
                    self.__sysTemp[i][9].update({j: t})
                    self.__sysTemp[j][9].update({i: t})
                else:
                    self.__sysTemp[i][7].update({j: t})
                    self.__sysTemp[j][7].update({i: t})
                self.__sysTemp[i][8].update({j: Em})
                self.__sysTemp[j][8].update({i: Em})

        if self.verbose:
            print(
                "{te}unnel coupling ".format(te='2e- t' if twoe else 'T')
                + str(t) + " and mutual capacitance " + str(Em) 
                + " added between dots: " + str(dotnames) + '.'
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
        self.objects[name] = {
            'couplings': {},
            'level': level,
            'isLead': True,
            'numCouplings': 0,
            'name': name,
        }
        # Add lead to internal list of system objects
        self.__sysTemp.append([name,0,False,True,level,'none',1,{},{},{},0])
        # Create couplings dict. so 'leads' may be searched like dots
        for i,dot in enumerate(dots):
            self.add_coupling(0,t[i],(name,dot),twoe=False)
        if self.verbose:
            print(
                "Lead with chemical potential " + str(level) 
                + " added which couples to dots: " + str(dots) + "."
                )

    def delete(self, *names):
        """Delete dot/lead/coupling with [name].

        Adjusts .__sysTemp accordingly for deleted dot/lead.

        Parameters:
        names (string or tuple of strings): if string, deletes
            dot or lead with same name, if tuple of strings, deletes
            all couplings between named objects.
        """
        def del_coupling(*names):
            """Deletes coupling between objects in names."""
            if np.isscalar(names) or len(names)==1:
                name = self.__cast_as_sclr(names)
                for n in [n for n in self.objects if name in self.objects[n]['couplings']]:
                    del self.objects[n]['couplings'][name]
                    self.objects[n]['numCouplings'] -= 1
                for n in [n for n in self.objects if name in self.objects[n]['twoeCouplings']]:
                    del self.objects[n]['twoeCouplings'][name]
                    self.objects[n]['numCouplings'] -= 1
            else:
                for n1,n2 in permutations(*names,2):
                    del self.objects[n1]['couplings'][n2]
                    self.objects[n1]['numCouplings'] -= 1
            self.__tunnelingHam = {}
            self.__diagElements = {}

        if len(names) == 1:
            n = names[0]
            del self.objects[n]
            del_coupling(n)
            for i in [i for i,v in enumerate(self.__sysTemp) if v[0] == n]:
                del self.__sysTemp[i]
            # Dictionary of system states is no longer valid after object is removed
            self.states = {}
            self.__tunnelingHam = {}
            self.__diagElements = {}
        else:
            del_coupling(*names)
        self.__isFinalized = False

    def finalize(self):
        """Port system information from .__sysTemp to dictionary of numpy arrays in .__sys
        
        Translates system information to a dictionary of numpy arrays, so that vectorized
        calculations can be carried out afterwards.
        """
        l = self.__ndotseff # Number of eff. dots / leads in system
        # Convert .__sysTemp to numpy array of all objects
        sys = np.array(self.__sysTemp,dtype=object) 
        # First translate each object's properties into numpy array
        self.__sys.update({
            'name': np.array(sys[:,0],dtype=str),
            'Ec': np.array(sys[:,1],dtype=float),
            'isSC': np.array(sys[:,2],dtype=bool),
            'isLead': np.array(sys[:,3],dtype=bool),
            'orb': np.array(sys[:,4],dtype=list),
            'spin': np.array(sys[:,5],dtype=str),
            'deg': np.array(sys[:,6],dtype=int),
            'Em': np.zeros((l,l)),
            't': np.zeros((l,l)),
            'twoet': np.zeros((l,l)),
            'u': np.array(sys[:,10],dtype=float)
        })
        # Appoint spin up/down to each quasi-dot, excluding SC islands
        for j,dt in enumerate(sys[:,5]):
            if dt == 'spin' and not sys[j,2]:
                if j%2 == 0:
                    self.__sys['spin'][j] = 'down'
                else:
                    self.__sys['spin'][j] = 'up'
        # Next, translate couplings into numpy array in .__sys as well
        for i,j in combinations(range(l),2):
            t = sys[i][7]
            Em = sys[i][8]
            self.__sys['Em'][i,j] = Em[j] if j in Em else 0
            self.__sys['t'][i,j] = t[j] if j in t else 0
        # Finally, symmetrize the 't' and 'Em' arrays:
        self.__sys['t'] = symmetrize(self.__sys['t'])
        self.__sys['Em'] = symmetrize(self.__sys['Em'])
        self.__isFinalized = True
        
    def get_states(self, N):
        """Generate all possible charge/orbital states for given total charge N.

        Parameters:
        N (int): If system has no leads: Number of charges in system.
            Otherwise: Maximum number of charges per dot.
            Leads may have nleads*ndots*N charges

        Returns:
        numpy.array: Contains state of each (quasi-)dot/lead in same order as in
            self.__sys.
        """
        if not self.__isFinalized:
            self.finalize()
        sys = self.__sys 
        nObjs = self.__ndotseff
        areLeads = self.nleads > 0
        areDegenerateDots = self.__ndotseff != self.ndots
        nMax = N*self.ndots*self.nleads if self.nleads > 0 else N
        # List of all indices of 'quasi'-dots comprising degenerate dots
        degIndices = np.array([
            [i for i,v in enumerate(sys['name']) if sys['name'][i] == n]
            for n in set(sys['name']) if (sys['name'] == n).sum() > 1
        ])

        def EcNotEnforced(chgs,degIndices=degIndices):
            """Ensures degenerate dots don't have extra charge in degenerate orbital."""
            return any([
                any([abs(d[0]-d[1]) > 1
                for d in combinations(chgs[inds],2)
                ])
                for inds in degIndices
                ])

        # Assign positive numbers to spin up and negative to spin down.
        spin = []
        for i,s in enumerate(sys['spin']):
            if s == 'down':
                spin.append(-1)
            elif s == 'up':
                spin.append(1)
            else:
                spin.append(0)

        def dotstates(charges):
            """Returns all possible dot/lead (of index i) states for given charge."""
            # First index corresponds to charge, second to quantum number of quasi-dot
            # and third to whether or not dot is spinful.
            return [
                [[c,j+1,1] for j in range(sys['deg'][i]//2)]
                + [[c,-j-1,1] for j in range(sys['deg'][i]//2)]
                if sys['isSC'][i] and charges[i]%2 and sys['spin'][i] == 'spin'
                else [[c,j,0] for j in range(sys['deg'][i])]
                if sys['isSC'][i] and charges[i]%2
                else [[c,0,1]]
                if sys['isSC'][i] and sys['spin'][i] == 'spin'
                else [[c,spin[i],abs(spin[i])]]
                for i,c in enumerate(charges)
                ]

        def dotChargeIsNotN(charges,N):
            """Returns whether or not dot charges are N, ignoring leads."""
            return np.any(charges[sys['isLead']==False] > N)

        def partitions(n,k,NInput,l=0,kc=None):
            """Yields possible charge/orbital states.

            Finds all possible partitions of n into k integers with minimum
            parition size l. kc is used for the recursive part of the algorithm.
            Augments this algorithm by generating all unique permutations of 
            results, and checking which are physical for the system at hand.
            """
            if kc == None:
                kc = k
            if kc < 1:
                # If no integers are left, we are done.
                return
            if kc == 1:
                if n >= l:
                    yield (n,)
                # If only one integer is left, and the remainder is larger than
                # the minimum size l, we are done.
                return
            for i in range(l,n+1):
                # Call partitions with l=i to ensure that no tuple is yielded twice,
                # as this enforces i being the largest integer
                for r in partitions(n-i,k,NInput,l=i,kc=kc-1):
                    s = tuple(r) + (i,)
                    # If len(s) == n, we have an (undordered) charge state
                    if len(s) == k:
                        # Find all ordered charge states from unordered states
                        for p in [np.array(p) for p in multiset_permutations(s)]:
                            # If degenerate dots exist, enforce charging energy.
                            if areDegenerateDots and EcNotEnforced(p):
                                continue
                            # Enforce that dots can only have N electrons, while leads
                            # can have ndots*nleads*N electrons:
                            if areLeads and dotChargeIsNotN(p,N):
                                continue
                            for state in product(*dotstates(p)):
                                yield np.array(state)
                        continue
                    yield s

        ti = time.perf_counter()
        self.states[N] = np.array([chgs for chgs in partitions(nMax,nObjs,N)])
        if self.verbose:
            print("Time to generate " + str(len(self.states[N][:,0,0]))
            + " system states was " + str(time.perf_counter() - ti) + 's')


        return self.states[N]
        
    def dimension(self, N):
        """Return dimension (int) of Hilbert space for a given number of charges N in the system."""
        if N not in self.states:
            self.get_states(N)
        return len(self.states[N][:,0,0])

    def onsite(self,states,gates,N=None):
        """Vectorized function to find on-site energy of system states.

        Given numpy array with system states listed along one axis,
        calculates energy of each state and returns it as a vector.

        Parameters:
        states (ndarray): Array of system states. Should have shape
            (#states,#effective system objects,2)
        gates (dict): Reduced gate voltages for each object in system.
            note: Leads should have 'gate' voltage 0 due to limitations
            in simulation. 

        Returns:
        E (ndarray): numpy array of onsite energy corresponding to each state.
        """
        if not self.__isFinalized:
            self.finalize()
        sys = self.__sys
        numStates = len(states[...,0])
        # Compute total charging energy
        EcTot = np.zeros(numStates)
        for n in gates:
            Ec = sys['Ec'][sys['name']==n][0]
            # Must sum degenerate charges on quasi-dots before squaring, since
            # they correspond to the same 'real' dot
            ns = np.sum(states[:,sys['name']==n,0],axis=1) - gates[n]
            EcTot += Ec*ns**2
        if N == None or N not in self.__diagElements:
            # Compute all diagonal 'expectation values' of Em to find mutual capacitance energy.
            EmTot = np.einsum('ji,ij->j',states[...,0], sys['Em'] @ states[...,0].T)/2
            # Compute orbital energy of SC quasiparticles then of electrons on normal dots.
            Eorb = np.einsum(
                'i,ij',
                np.array(np.where(sys['isSC'],sys['orb'],0),dtype=float),
                states[...,0].T%2
                )
            # Add orbital energies of normal dots:
            Eorb +=np.sum(
                [
                    np.take(o,states[:,i,0],mode='clip')
                    for i,o in enumerate(sys['orb'][sys['isSC']==False])
                ],
                axis = 0
                )
            Eother = EmTot + Eorb
            self.__diagElements[N] = Eother
        else:
            Eother = self.__diagElements[N]
        return EcTot + Eother

    def are_neighbours(self, states, N = None, hamiltonian=False):
        """Find which in set of states are nearest neighbours.
        
        For each pair of states in an array of states, determines
        which differ by only a single charge transfer, without any hopping
        of electrons from one quasiparticle state to another.
        
        Parameters:
        states (ndarray): Array where each row is a possible state

        Keyword Arguments:
        hamiltonian (bool): Whether or not to return tunneling matrix
        
        Returns:
        (ndarray): Array with size equal to the number of input states squared,
            with True in all matrix elements where the corresponding states are 
            nearest neighbours, and False in all others (hamiltonian == True), or
            the tunneling Hamiltonian given couplings in .__sys['t']
            (hamiltonian == False).
        """
        if hamiltonian:
            if N == None:
                raise Exception(
                    "Total charge must be specified to store Hamiltonian in "
                    + ".__tunnelingHam for later use!"
                )
            elif N in self.__tunnelingHam:
                return self.__tunnelingHam[N]
        if not self.__isFinalized:
            self.finalize()
        sys = self.__sys
        nStates = len(states[:,0,0])
        ti = time.perf_counter()
        # Generate array (numStates,numStates,len(state)) of differences between
        # each combination of states.
        diffs = np.array([states[i,...] - states for i in range(nStates)])
        # Find states where exactly 1 particle hopped, assumes states have
        # conserved total charge:
        nbrs = np.sum(np.abs(diffs),axis=2)[...,0] == 2
        # Ensure no quasiparticle hopping has occurred as well:
        nbrs *= np.any((diffs[...,0] == 0)*(diffs[...,1] != 0),axis=2) == False
        # Calculate the total spin of each state
        spins = states[...,0]*states[...,1]*states[...,2]
        # Account for fact that SC islands can have max |spin| = 1
        spins[(spins > 0)*sys['isSC']] = 1
        spins[(spins < 0)*sys['isSC']] = -1
        totSpin = np.sum(spins,axis=1)
        # Calculate spin differences between each state combo
        spinDiffs = np.abs([totSpin[i] - totSpin for i in range(nStates)])
        # Make sure all nearest neighbours conserve spin, or obtain
        # their spin from a spinless source
        nbrs *= (spinDiffs == 0) + (spinDiffs == 1)
        # Select vectors of charge differences for only nearest neighbours:
        nbrDiffs = diffs[nbrs,:,0]
        # Calculate array of obj. indices where charge was transferred
        objInds = np.nonzero(nbrDiffs)[1].reshape((len(nbrDiffs[:,0]),2)).T
        # Flatten indices to match size of .__sys['t'] (tunnel coupling matrix)
        objInds = np.ravel_multi_index(objInds,sys['t'].shape)
        # Find tunnel coupling magnitude for each nearest neighbour
        ts = np.take(sys['t'],objInds)
        ham = np.zeros((nStates,nStates))
        ham[nbrs] = ts
        if self.verbose:
            print('Time to generate all state combos was ' + str(time.perf_counter()-ti) + 's.')
        if not hamiltonian:
            nbrs[ham == 0] = False
            return nbrs
        self.__tunnelingHam[N] = ham
        return ham

    def get_hamiltonian(self,N,gates):
        """Generate the system Hamiltonian for charge N & a single set of gate voltages."""
        if not self.__isFinalized:
            self.finalize()
        if N not in self.states:
            self.get_states(N)
        states = self.states[N]
        ham = np.diag(self.onsite(states,gates,N=N))
        ham += self.are_neighbours(states,N=N,hamiltonian=True)
        return ham

    def __choose_states(self,N):
        """Returns system states for a given choice of total charge N."""
        if N == None:
            if len(self.states) > 0:
                return self.states[max([k for k in self.states])]
            else:
                raise Exception(
                    "System states must be already generated or total "
                    + "charge must be specified to get number operator."""
                    )
        else:
            if N in self.states:
                return self.states[N]
            else:
                return self.get_states(N)

    def state_from_index(self,i,N=None):
        """Returns state vector given its flattened index in Hilbert space for N charges."""
        states = self.__choose_states(N)
        return states[i,...]

    def get_num_op(self,name,N=None):
        """Returns number operator for object 'name' for system with total charge N."""
        if not self.__isFinalized:
            self.finalize()
        states = self.__choose_states(N)
        return np.diag(np.sum(states[:,self.__sys['name']==name,0],axis=1))

    def cp_stability_diagram(
        self,probeName,leverArms,gates,
        N=None,T=0,nLevels=None,tempTol=0.1,sparse=False,removeJumps=False,flipAxes=False,
        **plotparams):
        """Generate and plot parametric capacitance stability diagram.
        
        Given the defined qdsystem and total charge N (per dot if leads exist),
        solves for the system ground state at each value of 2 swept gate voltages,
        and produces a colormap of the parametric capacitance, the GS energy, and the
        number expectation values of the two objects whose voltages are swept.
        
        Parameters:
        
        Keyword Arguments:
        
        """
        if not self.__isFinalized:
            self.finalize()
        # Generate system states if necessary
        self.__choose_states(N)
        # Pick out the 'dot' names for the two voltages that are varying:
        gvar = np.array([n for n,g in gates.items() if not np.isscalar(g) and len(g) > 1])
        if len(gvar) != 2:
            raise Exception('Two gates must be swept over at a time!')
        if probeName not in gvar:
            raise Exception('Parametric capacitance can only be probed from a gate which varies!')
        coupledDot = gvar[gvar != probeName][0]
        # Generate number operators for two gates with lever arms
        numOp1 = self.get_num_op(probeName,N)
        numOp2 = self.get_num_op(coupledDot,N)
        # If temperature != 0, make sure more than one eigenvalue of hamiltonian is obtained
        if T != 0 and nLevels == None:
            nLevels = int(self.dimension(N)/10)
        elif T == 0:
            nLevels = 1
        # Initialize matrix of number expectation values.
        ns = np.zeros((len(gates[probeName]),len(gates[coupledDot]),2))
        energies = np.zeros((len(gates[probeName]),len(gates[coupledDot])))
        for i,g0 in enumerate(gates[probeName]):
            ti = time.perf_counter()
            for j,g1 in enumerate(gates[coupledDot]):
                # Calculate and generate Hamiltonian for each voltage combination
                gs = dict(gates)
                gs.update({probeName: g0+leverArms[0]*g1, coupledDot: g1+leverArms[1]*g0})
                h = self.get_hamiltonian(N,gs)
                if sparse:
                    e,v = eigsh(h,k=nLevels,which='SM')
                    v0 = v
                    e0 = e
                else:
                    e,v = eigh(h,eigvals=(0,nLevels-1))
                    v0 = v[:,0:nLevels]
                    e0 = e[0:nLevels]
                if T != 0:
                    # Calculate partition function:
                    Z = np.sum(np.exp(-e0/T))
                    if Z == 0:
                        raise Exception(
                            "Partition function is 0. Use higher kT or "
                            + "approximate system as 0 temperature.")
                    if np.exp(-e0[-1]/T)/Z > tempTol:
                        raise Exception(
                            "At temperature kT = {T}, more states ".format(T=T)
                            + "than nLevels = {n} should be included.".format(n=nLevels)
                            )
                    # Calculate energy expectation value:
                    energies[i,j] = np.sum(e0*np.exp(-e0/T))/Z
                    # Calculate number expectation values:
                    ns[i,j,0] = np.sum([
                        (v0[:,i] @ numOp1 @ v0[:,i])*np.exp(-e0[i]/T)/Z
                        for i in range(nLevels)
                        ])
                    ns[i,j,1] = np.sum([
                        (v0[:,i] @ numOp2 @ v0[:,i])*np.exp(-e0[i]/T)/Z
                        for i in range(nLevels)
                        ])
                else:
                    energies[i,j] = e0
                    ns[i,j,0] = v0.T @ numOp1 @ v0
                    ns[i,j,1] = v0.T @ numOp2 @ v0
            if self.verbose:
                print("Finished row {row}/{totRows} in: {t}s.\r".format(
                    row=i+1,totRows=len(gates[probeName]),t=time.perf_counter()-ti))
        spacing1 = abs(gates[probeName][1] - gates[probeName][0])
        spacing2 = abs(gates[coupledDot][1] - gates[coupledDot][0])
        cp = (
            1*np.diff(ns[...,0],n=1,axis=1)[0:-1,:]/spacing1
            + leverArms[0]**2*np.diff(ns[...,1],n=1,axis=0)[:,0:-1]/spacing2
            + leverArms[0]*np.diff(ns[...,1],n=1,axis=1)[0:-1,:]/spacing1
            + leverArms[0]*np.diff(ns[...,0],n=1,axis=0)[:,0:-1]/spacing2
        )
        if removeJumps:
            print(np.std(cp))
            cp[cp - np.mean(cp) > 2*np.std(cp)] = 0
        if flipAxes:
            xAxis = gates[coupledDot]
            yAxis = gates[probeName]
            ns0 = ns[...,0]
            ns1 = ns[...,1]
        else:
            xAxis = gates[probeName]
            yAxis = gates[coupledDot]
            cp = cp.T
            energies = energies.T
            ns0 = ns[...,0].T
            ns1 = ns[...,1].T
        fig,axs = plt.subplots(2,2,sharex=True,sharey=True)
        ax = axs[0,0]
        im = ax.pcolormesh(xAxis,yAxis,cp,**plotparams)
        fig.colorbar(im,ax=ax,label='C_p')
        ax.set_xlabel(gvar[0] + ' reduced voltage')
        ax.set_ylabel(gvar[1] + ' reduced voltage')
        ax = axs[0,1]
        im = ax.pcolormesh(xAxis,yAxis,energies,**plotparams)
        fig.colorbar(im,ax=ax,label='GS Energy (rel. units)')
        ax.set_xlabel(gvar[0] + ' reduced voltage')
        ax.set_ylabel(gvar[1] + ' reduced voltage')
        ax = axs[1,0]
        im = ax.pcolormesh(xAxis,yAxis,ns0,**plotparams)
        fig.colorbar(im,ax=ax,label='<n> (probed dot)')
        ax.set_xlabel(gvar[0] + ' reduced voltage')
        ax.set_ylabel(gvar[1] + ' reduced voltage')
        ax = axs[1,1]
        im = ax.pcolormesh(xAxis,yAxis,ns1,**plotparams)
        fig.colorbar(im,ax=ax,label='<n> (other dot)')
        ax.set_xlabel(gvar[0] + ' reduced voltage')
        ax.set_ylabel(gvar[1] + ' reduced voltage')
        plt.show()
        
def symmetrize(mat):
    """Symmetrizes a numpy array like object and halves its entries"""
    return (mat + mat.T - np.diag(mat.diagonal()))/2

def main():
    print("Calling main() yields a stability diagram of a N-SC DQD "
        + "where the SC has a spinful subgap state, and the N dot "
        + "has spin degeneracy 2. This will take a while...")
    N = 2
    system = DotSystem(verbose=True)
    #system.add_dot(100,degeneracy=1,orbitals=100,isSC = False)
    system.add_dot(300,name='dot0',degeneracy=1,orbitals=0,spin=True,isSC=False)
    system.add_dot(100,name='dot1',degeneracy=1,orbitals=80,spin=True,isSC=True,
        uv=[np.sqrt(0.2),np.sqrt(0.8)])
    system.add_lead(['dot0'],[10],name='lead0',level=0)
    system.add_lead(['dot1'],[10],name='lead1',level=0)
    system.add_coupling(60,20,['dot0','dot1'],twoe=False)
    print(system)
    #system.add_dot(400,name='SCdot',degeneracy=1000,orbitals=200,isSC=True)
    system.get_states(N)
    npoints = 100
    gates = {'dot0': np.linspace(0,N,npoints), 'dot1': np.linspace(0,N+1,npoints)}
    leverArms = [0.01,0.5]
    system.cp_stability_diagram(
        'dot0',leverArms,gates,N=N,
        T=0,sparse=False,removeJumps=False,flipAxes=False,
        cmap='seismic_r')
    
if __name__ == '__main__':
    main()