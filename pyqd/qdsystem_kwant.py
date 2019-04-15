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
        self.dots = {}                     # Dictionary of dots & their parameters
        self.couplings = {}                # Dictionary of couplings/capacitances between dots
        self.leads = {}                    # Dictionary of leads and which dots they attach to
        self.sys = kwant.builder.Builder() # kwant system to be constructed from dots&couplings
        self.print = print                 # Suppresses printing function results if False
    
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
    def nleads(self):
        """Number of leads in system."""
        return len(self.leads)

    @property
    def ncouplings(self):
        """Number of tunnel couplings (not incl. those to leads)."""
        return len(self.couplings)

    def add_dot(self, Ec, name=None, degeneracies=[0,0], orbEnergies=[0,0]):    
        """Add a quantum dot to the system.

        Keyword Arguments:
        name (str): Key name (in self.dots) of the dot
        Ec (float): Charging energy of the dot
        degeneracies (2x1 list, int): Orbital or spin degeneracy for
            even/odd parity, respectively.
        orbEnergies (2x1 list, float): Orbital addition energy or SC
            gap for even/odd parity charge state, respectively.

        Default settings:
            Dot with no degeneracy or confinement energy is added.
        """
        if name is None: name = 'dot' + str(self.ndots)
        index = self.ndots + self.nleads
        self.dots[name] = {
            'Ec': Ec,
            'degeneracies': degeneracies,
            'orbEnergies': orbEnergies,
            'numCouplings': 0,
            'type': 'dot',
            'index': index
            }
        if self.print:
            print("Dot added with name: " + str(name) + ".")

    def add_coupling(self, Em, t, dotnames, name=None):
        """Add tunnel coupling or mutual capacitance between two dots.
        
        Parameters:
        Em (float): Mutual capacitance energy (in ueV) between the two dots.
            Em = (e^2/Cm) / (C1*C2/Cm - 1)
        t (float): Tunnel coupling between the two dots (in ueV).
        dotnames (list of str): List of names of dots to be coupled.

        Keyword Arguments:
        name (str): Key word for coupling in self.couplings.
        lead (bool): Whether or not coupling is to a lead.
        """
        if name is None: name = dotnames[0] + '_' + dotnames[1]
        if len(dotnames) > 2:
            raise Exception('Couplings can only be defined between two dots at a time!')
        for dot in (dot for dot in dotnames if dot not in self.dots):
            raise Exception(dot + ' is not a defined dot!')
        for c in self.couplings:
            if all([dn in c['dots'] for dn in dotnames]):
                del self.couplings[c]
                warnings.warn('Coupling already exists between ' + str(dotnames[0]) + ' and ' + str(dotnames[1])
                    + "! Overwriting with new coupling parameters.")
        self.couplings[name] = {
            'Em': Em,
            't': t,
            'dots': dotnames,
            'lead': lead
            }
        for dot in dotnames: self.dots[dot]['num_couplings'] += 1
        if self.print:
            print("Tunnel coupling and capacitance '" + name + "' added between dots: '" + str(dotnames) + '.')

    def add_lead(self, dots, couplings, name=None, level=0):
        """Add a lead to the system.

        Models lead as a quantum dot with 0 charging energy and many
        electrons, plus a tunnel coupling to certain dots.

        Parameters:
        dots (list): Dots (given by string keys) to which lead couples
        couplings (dict): Tunnel couplings to dots (in ueV), with each coupled
            dot identified by its keyword in self.dots.
        
        Keyword Arguments:
        name (str): Key to access entry for this lead in self.leads
        level (float): Chemical potential (in ueV) of electrons in this lead
        """
        if name == None: name = 'lead' + str(self.nleads)
        index = self.ndots + self.nleads
        self.leads[name] = {
            't': couplings,
            'couplings': dots,
            'level': level,
            'type': 'lead',  # for distinguishing from regular dots while sending system to kwant
            'index': index
        }
        if self.print:
            print("Lead with chemical potential " + str(level) + "added which couples to dots: " + str(dots))

    def state_ravel(self, dot, state):
        """Converts kwant lattice position of dot state to charge/orbital state.

        ## PG. 92-94 of MSc Logbook details derivation of this expression ##
        Given the position along a given dot's kwant lattice axis, the charge
        state and the orbital state of the dot are returned.

        Parameters:
        dot (str): keyword of dot in self.dots whose state is being considered.
        state (int): Position of dot state along its kwant lattice axis.

        Returns:
        charge (int): Charge occupation of dot.
        orb (int): index of orbital state of dot. 
        """
        d = self.dots[dot]['degeneracies']
        charge = 2*(state // (d[0]+d[1])) + min([(state % (d[0]+d[1])) // d[0], 1])
        orb = state % (d[0]+d[1]) - (charge % 2)*d[0]
        return charge, orb

    def state_unravel(self, dot, state):
        """Convert charge/orbital state to kwant lattice position.

        ## PG. 92-94 of MSc Logbook details derivation of this expression ##
        Given the charge and orbital state of a dot, returns the index
        of that state in the corresponding kwant lattice.

        Parameters:
        dot (str): keyword of dot in self.dots whose state is being considered.
        state (2x1 list of ints): List with first index being charge number,
            second index being the orbital index (0,1,...,degeneracy-1)

        Returns:
        int: Position index of dot state on that dot's kwant lattice axis.
        """
        d = self.dots[dot]['degeneracies']
        charge = state[0]
        orb = state[1]
        return (charge//2)*(d[0]+d[1]) + (charge%2)*d[0] + orb    
        
    def dot_states(self, N, dictform=False):
        """Generate all possible charge/orbital states for given total charge N.

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
            N electrons, and each lead can contain ALL electrons from each dot."""
            dotstates = [None]*(self.ndots + self.nleads)
            for dot, dotname in self.dots.items():
                degen = dot['degeneracies']
                dotstates[dot['index']] = list(range(0, self.state_unravel(dotname, [N, degen[N%2]-1])))
            for lead in self.leads.values():
                dotstates[lead['index']] = list(range(0, self.ndots*N + 1))
            for state in product(*dotstates):
                yield state

        def is_possible_charge_state(state, N):
            """Checks if a charge state, given as a tuple, is allowable for fixed total N."""
            totalCharge = 0
            for dot, dotname in self.dots.items():
                totalCharge += self.state_ravel(dotname, state[dot['index']])[0]
            for lead in self.leads.values():
                totalCharge += state[lead['index']]
            if (totalCharge != N and self.nleads == 0) or (totalCharge != N * self.ndots and self.nleads != 0):
                return False
            return True

        def state_formatter(state):
            if dictform:
                return {k: [state[v['index']], self.state_ravel(k, state[v['index']])[1]] for (k,v) in self.dots.items()}
            else:
                return state

        for state in unfixed_charge_states(N):
            if is_possible_charge_state(state, N): yield state_formatter(state)
    
    def dimension(self, N):
        """Return dimension (int) of Hilbert space for a given number of charges N in the system."""
        return len(list(self.dot_states(N)))
                 
    def full_lattice(self):
        """Return kwant lattice object given self.dots and self.couplings"""
        numAxes = self.ndots + self.nleads
        if self.print: ti = time.clock()
        lat = kwant.lattice.Monatomic(np.identity(numAxes))
        if self.print: print('Time to generate lattice was: ' + str(time.clock() - ti) + 's.')
        return lat

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

        # Generate kwant lattice given the current number of dots and leads in DotSystem
        self.sys.lat = self.full_lattice()

        # Add diagonal Hamiltonian terms, i.e. onsite energies
        def onsite(site, gates):
            pos = site.pos  # State vector
            stateEnergy = 0 # Initialize energy of state, to be returned to kwant
            ns = {}         # Dictionary for storing charge state of each dot
            for dotname, dot in self.dots.items():
                dotcharge = self.state_ravel(dotname, pos[dot['index']])[0]
                ns[dotname] = dotcharge - gates[dotname]
                # Add parity dependent orbital/gap energy if one is present
                stateEnergy += dot['orbEnergies'][dotcharge % 2]
                stateEnergy += dot['Ec'] * ns[dotname] ** 2 # Add charging energy
            for lead in self.leads.values():
                # Add chemical potential of all electrons in leads
                stateEnergy += pos[lead['index']] * lead['level']
            # Add mutual capacitance energy for all 'real' dots (i.e. excluding leads)
            dot_couplings_gen = [c for c in self.couplings.values() if c['lead'] == False]
            for c in dot_couplings_gen:
                stateEnergy += c['Em'] * ns[c['dots'][0]] * ns[c['dots'][1]]
            return stateEnergy
        if state_iterator is None:
            self.sys[(self.sys.lat(*states) for states in self.charge_states(N))] = onsite
        else:
            self.sys[(self.sys.lat(*states) for states in state_iterator)] = onsite

        #Add off-diagonal Hamiltonian terms, i.e. tunnel couplings

        ## STORE DICTFORM STATES AND KWANT FORM STATES IN A GLOBAL OBJECT, SO CHARGE OF A GIVEN STATE CAN ALWAYS BE REFERENCED.

        numAxes = len(self.sys.lat.prim_vecs)
        for coupling in self.couplings.values():
            # If coupling only has Ecm, then it was already accounted for with 'onsite' function
            if coupling['t'] == 0: continue

            dots = (self.dots[coupling['dots'][0]], self.dots[coupling['dots'][1]])
            indices = [dots[0]['index'], dots[1]['index']]
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
        dot = self.dots[dotname]

        numOp = np.zeros((dim,dim))
        for i in range(dim):
            pos = list(self.sys.sites[i].pos)
            if self.dots[dotname]['type'] == 'superconducting':
                numOp[i,i] = self.sc_state_mapper(dot, pos[dot['indices'][0]], want='charge')
            else:
                numOp[i,i] = sum(pos[dot['indices'][0]: dot['indices'][1] + 1])
        return numOp

    # Built-in analysis functions here?


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
                params = (gates,)

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