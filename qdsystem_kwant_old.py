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

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import kwant
import warnings
import time

from scipy.special import comb
from itertools import product
from sortedcontainers import SortedDict
from sympy.utilities.iterables import multiset_permutations

# To do:
# -Add tunnel couplings for SC dots
# -Add plotting features
# -Make comments/code more professional


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
    def __init__(self, print = True):
        """
        Constructor for DotSystem class.

        Parameters:
        print (bool): whether or not to print results of calling methods
        """
        self.dots = SortedDict()           # Indexable/ordered dictionary of dots & their parameters
        self.couplings = SortedDict()      # Dictionary of couplings/capacitances between dots
        self.leads = SortedDict()          # Dictionary of leads and which dots they attach to
        self.sys = kwant.builder.Builder() # kwant system to be constructed from dots&couplings
        self.print = print                 # Suppresses printing function results if False
    
    def __str__(self):
        """Print class type and attributes when qdsystem is called with print()."""
        output = "qdsystem class object with: \n"
        output += str(self.ndots) + " total dots, of which "
        output += str(self.ndotscharge) + " are normal dots, "
        output += str(self.ndotssc) + " are superconducting, and "
        output += str(self.ndotsfermionic) + " are fermionic."
        return output

    # Number of dots in system
    @property
    def ndots(self):
        return len(self.dots)

    @property
    def nleads(self):
        """Number of leads in system."""
        return len(self.leads)

    @property
    def ncouplings(self):
        """Number of tunnel couplings (not incl. those to leads)."""
        return len(self.couplings)
    
    @property
    def ndotscharge(self):
        """Number of charge type dots in system."""
        return len([dot for dot in self.dots.values() if dot['type'] == 'charge'])
    
    @property
    def ndotsfermionic(self):
        """Number of fermionic type dots in system."""
        return len([dot for dot in self.dots.values() if dot['type'] == 'fermionic'])

    @property
    def ndotssc(self):
        """Number of superconducting type dots in system."""
        return len([dot for dot in self.dots.values() if dot['type'] == 'superconducting'])

    @property
    def _quasidots(self):
        """For internal use. Adds adds leads to dots dictionary, since leads here are dots with 0 Ec."""
        return SortedDict({**self.dots, **self.leads})

    def add_dot(self, name='dot', Ec=100, levels=None, delta=0):    
        """Add a quantum dot to the system.

        Keyword Arguments:
        name (str): Key name (in self.dots) of the dot
        Ec (float): Charging energy of the dot
        delta (float): Superconducting gap of the dot
        levels (list): (delta == 0) Fermionic mode energies (fermionic dot)
            (delta != 0) Quasiparticle excitation energies (superconducting dot)
            Indicates a 'charge' type dot when it == None and delta == 0.
        numModes (int): len(levels), being the number of QP excitations considered
            when delta != 0, or number of fermionic modes for fermionic dots

        Default settings:
        delta != 0 and levels == None:
            A single quasiparticle state of energy delta is considered
        delta == 0 and levels == None:
            A normal dot with charge states (rather than fermionic modes)
            is considered
        """
        if name == 'dot':
            name = 'SC' + name + str(self.ndots) if delta != 0 else name + str(self.ndots)
        self.dots[name] = {
            'Ec': Ec,
            'delta': delta,
            'levels': levels,
            'num_couplings': 0
            }
        if delta == 0 and levels is None:
            self.dots[name]['type'] = 'charge'
            self.dots[name]['dimension'] = lambda n: 1
        elif delta != 0:
            self.dots[name]['levels'] = [delta] if levels is None else levels
            self.dots[name]['type'] = 'superconducting'
            self.dots[name]['dimension'] = lambda n: (n % 2)*len(levels) + (1 - n % 2)
        else:
            self.dots[name]['type'] = 'fermionic'
            self.dots[name]['dimension'] = lambda n: comb(len(levels), n)
        if self.dots[name]['levels'] != None: self.dots[name]['numModes'] = len(self.dots[name]['levels'])
        else: self.dots[name]['numModes'] = 0
        if self.print:
            print("Dot of '" + str(self.dots[name]['type']) + "' type added with name: " + str(name) + ".")

    def add_lead(self, dots, couplings, name = 'lead', level = 0):
        """Add a lead to the system.

        Models lead as a quantum dot with 0 charging energy and many
        electrons, plus a tunnel coupling to certain dots.

        Parameters:
        dots (list): Dots (given by string keys) to which lead couples
        couplings (list): Tunnel couplings to dots (in ueV), with order
            matching that of dots list
        
        Keyword Arguments:
        name (str): Key to access entry for this lead in self.leads
        level (float): Chemical potential (in ueV) of electrons in this lead
        """
        if name == 'lead': name += str(self.nleads)
        self.leads[name] = {
            't': couplings,
            'couplings': dots,
            'levels': level,
            'type': 'lead'  # for distinguishing from regular dots while sending system to kwant
        }
        for i, dot in enumerate(dots):
            self.add_coupling(0, couplings[i], dot1name = name, dot2name = dot, lead = True)
        if self.print:
            print("Lead with chemical potential " + str(level) + "added which couples to dots: " + str(dots))

    def add_coupling(self, Em, t, dot1name='dot0', dot2name='dot1', name=None, lead=False):
        """Add tunnel coupling or mutual capacitance between two dots.
        
        Parameters:
        Em (float): Mutual capacitance energy (in ueV) between the two dots.
            Em = (e^2/Cm) / (C1*C2/Cm - 1)
        t (float): Tunnel coupling between the two dots (in ueV).

        Keyword Arguments:
        dot1name (str): Key in self.dots for first dot in coupling.
        dot2name (str): Key in self.dots for second dot in coupling.
        name (str): Key for coupling in self.couplings.
        lead (bool): Whether or not coupling is to a lead.
        """
        defname = dot1name + '_' + dot2name
        if name is None: name = defname
        if dot1name not in self._quasidots: raise Exception(dot1name + ' is not a defined dot or lead!')
        if dot2name not in self._quasidots: raise Exception(dot2name + ' is not a defined dot or lead!')
        for c in self.couplings:
            if (dot1name in self.couplings[c]['dots']) and (dot2name in self.couplings[c]['dots']):
                warnings.warn("Coupling '" + c + "' already exists between " + dot1name + ' and ' + dot2name + '!\n'
                              'Overwriting with new input values.')
                del self.couplings[c]
        self.couplings[name] = {'Em': Em, 't': t, 'dots': [dot1name, dot2name], 'lead': lead}
        if self._quasidots[dot1name]['type'] != 'lead': self.dots[dot1name]['num_couplings'] += 1
        if self._quasidots[dot2name]['type'] != 'lead': self.dots[dot2name]['num_couplings'] += 1
        if self.print:
            print("Tunnel coupling and capacitance '" + name + "' added between dots '" + dot1name + "' and '" + dot2name + "'.")

    def sc_state_mapper(self, dot, want, N):
        """Return requested property of superconducting dot.

        Returns either the charge of a superconducting dot given the
        index of its state in the kwant lattice, its range of indices
        given the charge, or its energy given its state index.

        Parameters:
        dot (str): Key for superconducting dot in self.dots.
        want (str): Key for desired return of sc_state_mapper.
            May be 'energy', 'charge', or 'index' when the dot
            state's corresponding energy, charge state, or list
            of possible indices is desired.
        N (int): Index of kwant state vector when want is 'energy' or 'charge',
            or charge on dot when want is 'index'

        Returns:
        If want is 'energy' or 'charge':
            int: Energy of electron level or charge on dot
        If want is 'index':
            list: 1x2 list of first and last kwant lattice indices
                on the superconducting dot's axis corresponding
                to the charge state given by N.
        """
        nqp = dot['numModes']
        if want == 'energy':
            return dot['delta'] + dot['levels'][int(N % (nqp + 1)) - 1] if N % (nqp + 1) else 0
        elif want == 'charge':
            m = N + 1
            return 2*(N // (nqp + 1)) + min([N % (nqp + 1), 2]) if nqp > 1 else N
        elif want == 'index':
            startIndex = (N//2) * (nqp+1) + N%2
            interval   = (N%2) * (nqp-1)
            return list(range(startIndex, startIndex + interval + 1))
        else:
            raise Exception("Must input 'charge, 'index', or 'energy' for 'want' keyword.")        
        
    def charge_states(self, N, dictform=False):
        """Generate all possible charge states for given total charge N.

        Parameters:
        N (int): If system has no leads:
            Number of charges in system.
            Otherwise:
            Maximum number of charges per dot.
        
        Keyword Arguments:
        dictform (bool): Whether or not to yield charge states
            as SortedDict or as tuple.

        Yields:
        if dictform == True:
            SortedDict: Charge of each dot stored via same keyword
                used in self.dots
        if dictform == False:
            tuple: tuple of each dot's charge in same order as that
                determined by self.dots' (SortedDict) ordering.
        """

        def unfixed_charge_states(N):
            """Generates all charge states without a fixed total charge, wherein each dot may contain up to
            N electrons. """
            dotstates = tuple()
            for dot in self._quasidots.values():
                if   dot['type'] == 'charge': dotstates += (list(range(0, N + 1)),)
                elif dot['type'] == 'lead'  : dotstates += (list(range(0, self.ndots*N + 1)),)
                elif dot['type'] == 'fermionic':
                    dotstates += ([0,1],)*dot['numModes']
                elif dot['type'] == 'superconducting':
                    dotstates += (list(range(self.sc_state_mapper(dot, N, want = 'index')[-1] + 1)),)
                else:
                    raise Exception("Dot input that is not of one of the possible types: "
                                    "'charge', 'fermionic', or 'superconducting'.")
            for state in product(*dotstates):
                yield state

        def is_possible_charge_state(state, N):
            """Checks if a charge state, given as a tuple, is allowable for fixed total N."""
            if self.ndotssc == 0 and sum(state) > N and self.nleads == 0: return False
            totalCharge = 0
            for dot in self._quasidots.values():
                if 'indices' not in dot: self.indexer()
                dotCharge = sum(state[dot['indices'][0]: (dot['indices'][1] + 1)])
                if dot['type'] == 'fermionic':
                    if dotCharge > dot['numModes']: return False
                    if any([state[j] > 1 for j in range(dot['indices'][0], dot['indices'][1] + 1)]):
                        return False
                elif dot['type'] == 'superconducting':
                    dotCharge = self.sc_state_mapper(dot, state[dot['indices'][0]], want = 'charge')
                if dotCharge > N and dot['type'] != 'lead': return False
                totalCharge += dotCharge
            if (totalCharge != N and self.nleads == 0) or (totalCharge != N * self.ndots and self.nleads != 0):
                return False
            return True

        def state_formatter(state):
            if dictform:
                return SortedDict({k: state[v['indices'][0]: v['indices'][1] + 1] for (k,v) in self.dots.items()})
            else:
                return state

        for state in unfixed_charge_states(N):
            if is_possible_charge_state(state, N): yield state_formatter(state)
    
    def dimension(self, N):
        """Return dimension (int) of Hilbert space for a given number of charges N in the system."""
        return len(list(self.charge_states(N)))
                 
    def full_lattice(self):
        """Return kwant lattice object given self.dots and self.couplings"""
        numAxes = 0
        for dotname, dot in self._quasidots.items():
            numAxes += 1
            if dot['type'] == 'lead': continue   
            # Check if dot has couplings, else it is effectively isolated from the system
            if dot['num_couplings'] == 0:
                warnings.warn(dotname + ' does not have any tunnel couplings. It will be deleted.')
                del self.dots[dotname]
            elif dot['delta'] == 0:
                # Add 1 axis to lattice for charge / SC dots, and numModes axes to lattice for fermionic dots
                numAxes += max(dot['numModes'], 1) - 1
        if self.print: ti = time.clock()
        lat = kwant.lattice.Monatomic(np.identity(numAxes))
        if self.print: print('Time to generate lattice was: ' + str(time.clock() - ti) + 's.')
        return lat

    def indexer(self):
        """Update self.dots with each dot's indices in system state vector
        
        Adds 1x2 list containing first index corresponding to its dot levels
        and last index corresponding to its dot levels in lattice axes,
        stored in self.dots[dot]['indices']. Unless dot type is 'fermionic',
        the first and last index are the same.
        """
        i = 0
        for dot in self._quasidots.values():
            if dot['type'] != 'lead' and dot['levels'] != None and dot['delta'] == 0:
                dotLatticeDim = dot['numModes']
            else:
                dotLatticeDim = 1
            dot['indices'] = [i, i + dotLatticeDim - 1]
            i += dotLatticeDim

    def to_kwant(self, state_iterator=None, N=None, finalize=True):
        """Port information stored in DotSystem to kwant.

        Creates a kwant lattice for the system given its dots, then
        considers a subset of this lattice determined by the number of
        charges in the system, populating the kwant system with onsite
        energies and tunnel couplings determined by dots, leads and
        couplings.

        Keyword Arguments:
        state_iterator (iterator): Optionally an iterator defining a
            subset of the kwant lattice may be input as the system to be
            considered. Defaults to all possible charge states for a given N.
        N (int): Total charge in system if there are no leads (self.nleads == 0),
            or the maximum charge per dot if there are leads.
        finalize (bool): Whether or not to 'finalize' the kwant system (in that
            package's own syntax), once system is defined.
        """

        if self.nleads > 0 and N != None:
            warnings.warn("At least one dot has leads, so N cannot be fixed," 
                          " and will be interpreted as the maximum charge per dot.")

        self.sys.lat = self.full_lattice()
        # indexer cannot be called before full_lattice, else dots may be deleted which were already
        # accounted for in indexer
        self.indexer()

        # Add diagonal Hamiltonian terms, i.e. onsite energies
        def onsite(site, gates):
            pos = site.pos  # State vector
            stateEnergy = 0 # Energy of state, to be returned to kwant
            ns = {}         # Dictionary for storing charge state of each dot
            for dotname, dot in self._quasidots.items():
                if dot['type'] == 'superconducting':
                    ns[dotname] = self.sc_state_mapper(dot, pos[dot['indices'][0]], want='charge') - gates[dotname]
                    # Add quasiparticle energy if one is present
                    stateEnergy += self.sc_state_mapper(dot, pos[dot['indices'][0]], want='energy')
                elif dot['type'] == 'lead':
                    # Add chemical potential of all electrons in leads
                    stateEnergy += pos[dot['indices'][0]] * dot['levels']
                    ns[dotname] = 0
                else:
                    ns[dotname]  = sum(list(pos)[dot['indices'][0] : dot['indices'][1] + 1]) - gates[dotname]
                # Add dot energy resulting from non-zero charging energy
                if dot['type'] == 'superconducting' and stateEnergy > 99:
                    x = 1
                if dot['type'] != 'lead': stateEnergy += dot['Ec'] * ns[dotname] ** 2
                if dot['type'] == 'fermionic':
                    stateEnergy += np.dot(list(pos)[dot['indices'][0] : dot['indices'][1] + 1], dot['levels'])
            # Add mutual capacitance energy for all 'real' dots (i.e. excluding leads)
            gen = [c for c in self.couplings.values() if c['lead'] == False]
            for c in gen:
                stateEnergy += c['Em'] * ns[c['dots'][0]] * ns[c['dots'][1]]
            return stateEnergy
        if state_iterator is None:
            self.sys[(self.sys.lat(*states) for states in self.charge_states(N))] = onsite
        else:
            self.sys[(self.sys.lat(*states) for states in state_iterator)] = onsite

        #Add off-diagonal Hamiltonian terms, i.e. tunnel couplings
    
        numAxes = len(self.sys.lat.prim_vecs)
        for coupling in self.couplings.values():
            # If coupling only has Ecm, then it was already accounted for with 'onsite' function
            if coupling['t'] == 0: continue

            dots = (self._quasidots[coupling['dots'][0]], self._quasidots[coupling['dots'][1]])
            if dots[0]['type'] == 'superconducting' and dots[1]['type'] == 'superconducting':
                print('both dots are superconducting')
            elif dots[0]['type'] == 'superconducting' or dots[1]['type'] == 'superconducting':
                print('one of two dots is superconducting')
            else:
                indices = [dots[0]['indices'], dots[1]['indices']]
                for i, j in zip(range(indices[0][0], indices[0][1] + 1), range(indices[1][0], indices[1][1] + 1)):
                    vec = np.zeros(numAxes)
                    vec[i] = 1
                    vec[j] = -1
                    self.sys[kwant.builder.HoppingKind(vec, self.sys.lat)] = coupling['t']
        
        if finalize: self.sys = self.sys.finalized()
        
    def get_num_op(self, dotname, N):
        """Return number operator for a given dot.

        Parameters:
        dotname (str): key for dot in self.dots.
        N (int): Total charge in system if there are no leads,
            or maximum charge per dot if there are leads.
            Required because system's dimensionality is dependent on N.
        
        Returns:
        ndarray: Matrix form of the number operator for desired dot.
        """
        self.indexer()
        dim = self.dimension(N)
        dot = self._quasidots[dotname]

        numOp = np.zeros((dim,dim))
        for i in range(dim):
            pos = list(self.sys.sites[i].pos)
            if self._quasidots[dotname]['type'] == 'superconducting':
                numOp[i,i] = self.sc_state_mapper(dot, pos[dot['indices'][0]], want='charge')
            else:
                numOp[i,i] = sum(pos[dot['indices'][0]: dot['indices'][1] + 1])
        return numOp


"""
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
FOR TESTING
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
"""

def main():
    system = DotSystem(print=True)
    system.add_dot(levels = [0,0], delta = 100, Ec = 150, name = 'dot0')
    system.add_dot(Ec = 200, name = 'dot1')
    system.add_lead(['dot0'], [10], name = 'lead0', level = 0)
    system.add_lead(['dot1'], [10], name = 'lead1', level = 0)
    system.add_coupling(Em = 50, t = 10, dot1name = 'dot0', dot2name = 'dot1')
    
    N = 6
    npoints = 101
    print('System dimension is: ' + str(system.dimension(N)))

    if 1 == 1:    
        system.to_kwant(N = N)
        gatespace = {}
        for dot in system.dots: gatespace[dot] = np.linspace(0,N,npoints)
        spacing = gatespace['dot0'][1] - gatespace['dot0'][0]
        numOp0 = system.get_num_op('dot0', N)
        numOp1 = system.get_num_op('dot1', N)
        GSEnergy = np.zeros((npoints,npoints))
        number   = np.zeros((2, npoints, npoints))

        print('About to start diagonalizing a bunch of Hamiltonians...')
        for i, g1 in enumerate(gatespace['dot0']):
            for j, g2 in enumerate(gatespace['dot1']):
                gates = {}
                gates['dot0'] = g1
                gates['dot1'] = g2
                params = (gates, )

                if i == 0 and j == 0: ti = time.clock()
                H = system.sys.hamiltonian_submatrix(params)
                # try:
                #     E, vecs = sp.linalg.eigsh(H, k=1, which = 'SM')
                #     number[0,i,j] = np.dot(vecs.conj().T, np.dot(numOp0, vecs))[0][0]
                #     number[1,i,j] = np.dot(vecs.conj().T, np.dot(numOp1, vecs))[0][0]
                # except:
                # warnings.warn("Sparse diagonalization did not converge for " + str(g1) + ", " + str(g2) + ". Attempting exact   diagonalization.")
                Etot, vecstot = np.linalg.eigh(H)
                E = Etot[0]
                vecs = vecstot[:,0]
                number[0,i,j] = np.dot(vecs.conj().T, np.dot(numOp0, vecs))
                number[1,i,j] = np.dot(vecs.conj().T, np.dot(numOp1, vecs))

                if i == 0 and j == 2:
                    t = (time.clock() - ti)/3
                    print("Time to generate and diagonalize one Hamiltonian is: " + str(t) + "\n"
                        "(--> time for completion is: " + str(int((npoints * npoints - 1)*t//60)) + "m" + str(int((npoints * npoints - 1)*t%60)) + "s)")
                GSEnergy[i,j] = E
                if j % (npoints + 1) == 0: print('done up to: ' + str(i) + ' rows.\r')
        
        print('Done diagonalizing Hamiltonians.')

        # Quantum capacitance
        QC0 = np.zeros((npoints-1,npoints))
        QC1 = np.zeros((npoints, npoints-1))
        for j in range(npoints):
            QC0[:,j] = np.diff(number[0,j,:].T)/spacing
            QC1[j,:] = np.diff(number[1,:,j].T)/spacing

    system2 = DotSystem(print = True)
    system2.add_dot(levels = [0], delta = 100, Ec = 150, name = 'dot0')
    system2.add_dot(Ec = 200, name = 'dot1')
    system2.add_lead(['dot0'], [10], name = 'lead0', level = 0)
    system2.add_lead(['dot1'], [10], name = 'lead1', level = 0)
    system2.add_coupling(Em = 50, t = 10, dot1name = 'dot0', dot2name = 'dot1')

    print('System dimension is: ' + str(system2.dimension(N)))

    if 1 == 1:    
        system2.to_kwant(N = N)
        numOp02 = system2.get_num_op('dot0', N)
        numOp12 = system2.get_num_op('dot1', N)
        GSEnergy2 = np.zeros((npoints,npoints))
        number2   = np.zeros((2, npoints, npoints))

        print('About to start diagonalizing a bunch of Hamiltonians...')
        for i, g1 in enumerate(gatespace['dot0']):
            for j, g2 in enumerate(gatespace['dot1']):
                gates = {}
                gates['dot0'] = g1
                gates['dot1'] = g2
                params = (gates, )

                if i == 0 and j == 0: ti = time.clock()
                H = system2.sys.hamiltonian_submatrix(params)
                # try:
                #     E, vecs = sp.linalg.eigsh(H, k=1, which = 'SM')
                #     number[0,i,j] = np.dot(vecs.conj().T, np.dot(numOp0, vecs))[0][0]
                #     number[1,i,j] = np.dot(vecs.conj().T, np.dot(numOp1, vecs))[0][0]
                # except:
                # warnings.warn("Sparse diagonalization did not converge for " + str(g1) + ", " + str(g2) + ". Attempting exact   diagonalization.")
                Etot2, vecstot2 = np.linalg.eigh(H)
                E2 = Etot2[0]
                vecs2 = vecstot2[:,0]
                number2[0,i,j] = np.dot(vecs2.conj().T, np.dot(numOp02, vecs2))
                number2[1,i,j] = np.dot(vecs2.conj().T, np.dot(numOp12, vecs2))

                if i == 0 and j == 2:
                    t = (time.clock() - ti)/3
                    print("Time to generate and diagonalize one Hamiltonian is: " + str(t) + "\n"
                        "(--> time for completion is: " + str(int((npoints * npoints - 1)*t//60)) + "m" + str(int((npoints * npoints - 1)*t%60)) + "s)")
                GSEnergy2[i,j] = E
                if j % (npoints + 1) == 0: print('done up to: ' + str(i) + ' rows.\r')
        
        print('Done diagonalizing Hamiltonians.')

        # Quantum capacitance
        QC02 = np.zeros((npoints-1,npoints))
        QC12 = np.zeros((npoints, npoints-1))
        for j in range(npoints):
            QC02[:,j] = np.diff(number2[0,j,:].T)/spacing
            QC12[j,:] = np.diff(number2[1,:,j].T)/spacing

        fig, axs = plt.subplots(2,2, sharex = True, sharey = True)

        cmap = plt.get_cmap('viridis')
        # fig.subplots_adjust(wspace = 0.3) # Increase spacing between plots
        # fig.set_size_inches(10,10) # Increase size of output image (WxL)

        ax = axs[0,0]
        a = ax.pcolormesh(gatespace['dot0'], gatespace['dot1'], abs(GSEnergy.T - GSEnergy2.T), cmap = cmap)
        fig.colorbar(a, ax = ax)
        ax.set_xlabel('n_g1')
        ax.set_ylabel('n_g2')
        ax.set_title('GS Energy (ueV)')

        ax = axs[0,1]
        a = ax.pcolormesh(gatespace['dot0'], gatespace['dot1'], abs(((QC0[:,0:-1] + QC1[0:-1,:])/2) - ((QC02[:,0:-1] + QC12[0:-1,:])/2)), cmap = cmap)
        fig.colorbar(a, ax = ax)
        ax.set_xlabel('n_g1')
        ax.set_ylabel('n_g2')
        ax.set_title('Sum of Qc\'s')

        ax = axs[1,0]
        b = ax.pcolormesh(gatespace['dot0'], gatespace['dot1'], abs(number[0,:,:].T - number2[0,:,:].T), cmap = cmap)
        fig.colorbar(b, ax = ax)
        ax.set_xlabel('n_g1')
        ax.set_ylabel('n_g2')
        ax.set_title('d<n1>/dn_g1')

        ax = axs[1,1]
        c = ax.pcolormesh(gatespace['dot0'], gatespace['dot1'], abs(number[1,:,:].T - number2[1,:,:]), cmap = cmap)
        fig.colorbar(c, ax = ax)
        ax.set_xlabel('n_g1')
        ax.set_ylabel('n_g2')
        ax.set_title('d<n2>/dn_g2')

        plt.show()

        print(system.couplings)
    
    
if __name__ == '__main__':
    main()