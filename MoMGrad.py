import numpy as np
from scipy.linalg import block_diag
from scipy.linalg import expm

import pyquil.quil as pq
import pyquil.api as api
from pyquil.paulis import *
from pyquil.gates import *

from grove.pyvqe.vqe import VQE

import random


class MoMGrad1QB:
    """
    Simple minitbatched MoMGrad implementation for learning "quantum data".
    More Details of learning algo in the train() fxn
    """

    def __init__(self, q, rx, ry, rz):
        """
        Constructor function for 1qubit MoMGrad. Build a random 1qubt rotation
        and then use the train function to learn it.
        params:
        -------
                q: SyncConnection -> qvm connection to use for everything
                rx: float -> x pauli param (scaled up by a factor of pi)
                ry: float -> y pauli param (scaled up by a factor of pi)
                rz: float -> z pauli param (scaled up by a factor of pi)
        """

        self.qvm = q

        # LAZY, we only use this for expectation calculations and not actual
        # VQE stuff.
        from scipy.optimize import minimize
        self.vqe_inst = VQE(minimizer=minimize,
                            minimizer_kwargs={'method': 'nelder-mead'})

        # DON"T CHANGE BELOW UNLESS YOU KNOW WHAT YOU"RE DOING!
        # Initialize the indices and statistics for our Oscillators

        # Indices that our oscialltor will be simulated on
        self.oscialltor0 = [0, 1, 2]
        # Initial mu and sigma values for our oscillator
        self.oscialltor0_stats = [0.0, 0.95]

        self.oscialltor1 = [3, 4, 5]
        self.oscialltor1_stats = [0.0, 0.95]
        self.oscialltor2 = [6, 7, 8]
        self.oscialltor2_stats = [0.0, 0.95]

        # Our data qubits for the process we want to learn.
        self.datum_qubits = [9]

        # Our chosen learning rate.
        self.learning_rate = 0.3

        # The X,Y,Z params we want to leanr for our abitrary unitary.
        # Note if learning doesn't go so well there may have been an
        # internal phase overflow, so be careful....
        self.THETA1 = float(rx)  # 0.785
        self.THETA2 = float(ry)
        self.THETA3 = float(rz)  # 0.906

        # Lists to track our learning progress
        self.param_values = []
        self.fid_values = []

        # Number of samples to draw for each iteration of learning.
        self.n_samples = 400

    def make_final_qft_inv(self):
        """
        Helper function to give us the F^-1 matrix for our (qubit simulated) 
        harmonic oscillators

        """
        d = 7.
        indices = [-3, -2, -1, 0, 1, 2, 3]
        ret = np.identity(int(d), dtype=complex)

        def w_if(i, j):
            omega = np.e ** (((2.0*np.pi) * (0.0 + 1.0j)) / d)
            return omega ** (1.0 * (indices[i] * indices[j]))

        for i in range(len(indices)):
            for j in range(len(indices)):
                ret[i][j] = w_if(i, j)

        ret = block_diag(ret, np.asarray([[np.sqrt(d)]]))
        ret = ret * 1./np.sqrt(d)

        return ret

    def make_final_qft(self):
        """
        Same as above but makes F
        """
        U = make_final_qft_inv()
        return np.conj(U)

    def make_oscillator_gate(self, psi):
        """
        Gives a matrix to prepare an oscillator that has a given distribution
        specified by the coefficients in psi basically uses 1 iteration of
        shor (kinda)
        params:
        -------
                psi: list -> oscillator state probabilites

        returns:
        -------
                U: matrix -> unitary that prepares our oscillator 
        """
        psi = np.asarray(psi)
        zero = np.zeros_like(psi, dtype=complex)
        zero[0] = 1.0

        flag = True
        for i in range(len(psi)):
            if psi[i] != zero[i]:
                flag = False
        if flag:
            # psi is zero so return identity
            return np.identity(len(psi), dtype=complex)

        z = np.dot(zero, psi)

        theta = None
        if z == 0:
            theta = 0.0
        else:
            theta = np.arctan(z.imag / z.real)

        e_i_psi = (np.e ** ((0.0 - 1.0j) * theta)) * psi

        phi = e_i_psi - zero

        # WEIRD MINUS SIGN ISSUE !?!?!?
        U = np.identity(len(psi), dtype=complex) - \
            (2.0/np.vdot(phi, phi)) * np.outer(phi, np.conj(phi))

        U = (np.e ** ((0.0 + 1.0j) * theta)) * U
        return U

    def prep_oscillator(self, mu, sigma):
        """
        Gives probabilities for a 7dim oscillator with given mu and sigma

        params:
        -------
                mu: float -> center of the distribution for the oscillator
                sigma: float -> std of the distribution of the oscillator

        returns:
        --------
                psi: list -> list of state probabilities for the oscillator

        """

        d = 7.
        psi = [0 for i in range(int(d))]
        for i in range(len(psi)):
            big = np.e ** ((-(float(i) - (d - 1.) / 2. - mu)
                            ** 2.) / (4.0 * sigma ** 2.))
            psi[i] = big

        psi.append(0)
        psi = np.asarray(psi)

        denom = 0.0
        for i in range(len(psi)):
            denom += np.absolute(psi[i]) ** 2.

        if denom != 0:
            psi = psi * 1./np.sqrt(denom)

        return psi

    def get_measure_ham(self, indices):
        """
        Helper function so we can properly measure our simulated oscillators

        params:
        -------
                indices: list of length 3 -> qubit indices that make up an oscillator

        returns:
        --------
                overall_final: PauliSum -> hacky hamiltonian of paulis for 
                                           taking expectation values later
        """
        q0 = indices[0]
        q1 = indices[1]
        q2 = indices[2]

        overall_final = ZERO()

        overall_final += -0.5 * sI(q0) * sZ(q1) * sI(q2)
        overall_final += -0.5 * sI(q0) * sZ(q1) * sZ(q2)
        overall_final += -1.5 * sZ(q0) * sI(q1) * sI(q2)
        overall_final += -0.5 * sZ(q0) * sI(q1) * sZ(q2)
        overall_final += -0.5 * sZ(q0) * sZ(q1) * sI(q2)
        overall_final += 0.5 * sZ(q0) * sZ(q1) * sZ(q2)

        overall_final = overall_final.simplify()
        # print overall_final.simplify()

        return overall_final

    def make_bottom_phase_gate(self, lr, output_state):
        """
        Helper function to make the 'custom gate' corresponding to phase
        kicking the cost

        params:
        -------
                lr: float -> learning rate for our algorithm
                output_state: list -> output state to use in calculating cost

        returns:
        --------
                U: matrix -> unitary orresponding to gate

        """
        U = np.identity(2, dtype=complex) - \
            np.outer(output_state, np.conj(output_state))
        return expm((0.0 - 1.0j) * lr * U)

    def apply_para_gate(self, theta1, theta2, theta3):
        """
        Helper function to evaluate our paramterized state

        params:
        -------
                theta1: float -> X param
                theta2: float -> Y param
                theta3: float -> Z param

        returns:
        --------
                U: matrix -> result of:
          e^(-i/2 * theta1 * X) * e^(-i/2 * theta2 * Y) * e^(-i/2 * theta3 * Z)
        """
        zexp = expm((theta3 * (0.0 - 1.0j)/2.0) *
                    np.matrix([[1., 0.], [0., -1.]]))
        yexp = expm((theta2 * (0.0 - 1.0j)/2.0) *
                    np.matrix([[0, 0.0 - 1.0j], [0.0 + 1.0j, 0]]))
        xexp = expm((theta1 * (0.0 - 1.0j)/2.0) *
                    np.matrix([[0.0, 1.0], [1.0, 0.0]]))

        # return np.matmul(np.matmul(xexp, yexp), zexp)
        return np.matmul(zexp, np.matmul(yexp, xexp))

    def fidelity(self, theta1, theta2, theta3):
        """
        Helper function to calculate fidelity between two pure states
        defined with the paramaterized state above in apply_para_gate

        params:
        -------
                theta1: float -> X param
                theta2: float -> Y param
                theta3: float -> Z param

        returns:
        --------
                fidelity: float -> | <A | B > | ** 2
        """
        B = self.apply_para_gate(self.THETA1, self.THETA2, self.THETA3)
        A = self.apply_para_gate(theta1, theta2, theta3)

        a1 = np.dot(A, [(1.0 + 0.0j), (0.0 + 0.0j)])
        a2 = np.dot(B, [(1.0 + 0.0j), (0.0 + 0.0j)])

        return np.absolute(np.dot(a2, np.conj(a1))) ** 2

    def random_point_sample(self):
        """
        Helper function to generate uniform random samples from the surface of
        a sphere We return the following
        (
         |0>, 
         random Unitary,
         |random state>,
         |random state transformed by our black box we want to learn>
        )

        returns:
        --------
                zero_state: list -> bra-ket zero state

                U: matrix -> haar random matrix to transform zero state 

                initial_rand_state: list -> bra-ket random state to be fed 
                                            through our black box wwe want 
                                            to learn

                transformed state: list -> bra-ket of random state that 
                                            has been transformed by our 
                                            goal unitary

        """

        u = random.random()
        v = random.random()

        alpha = 2.0 * np.pi * u
        theta = np.arccos(2.0 * v - 1.0)

        theta /= 2.0

        U = np.matrix([[np.cos(theta),
                        -1.0 * np.sin(theta) * np.e ** ((0.0 - 1.0j) * alpha)],
                       [np.sin(theta) * np.e ** ((0.0 + 1.0j) * alpha),
                        np.cos(theta)]])

        tmp_prog = pq.Program().defgate("random_set", U)
        # if random.random() < 0.5:
        #   tmp_prog.inst(X(0))

        args = ('random_set', ) + tuple([0])
        tmp_prog.inst(args)

        # 0 state coef then 1 state coef.
        zero_state = np.asarray([1.0, 0.0])

        initial_rand_state = self.qvm.wavefunction(tmp_prog).amplitudes

        tmp_prog = pq.Program().defgate('random_set', U)
        tmp_prog += pq.Program().defgate('para_gate',
                                         self.apply_para_gate(self.THETA1,
                                                              self.THETA2,
                                                              self.THETA3))
        args = ('random_set', ) + tuple([0])
        tmp_prog.inst(args)

        args = ('para_gate', ) + tuple([0])
        tmp_prog.inst(args)

        transformed_state = self.qvm.wavefunction(tmp_prog).amplitudes

        return zero_state, U, initial_rand_state, transformed_state

    # oscialltor first qubit index second
    def make_controlled_pauli(self, pauli_type=None):
        """
        Helper function to create an oscillator controlled pauli gate

        params:
        -------
                pauli_type: str -> code for which pauli we're gonna make

        returns:
        --------
                U: matrix -> unitary corresponding to oscillator parameterized 
                                         pauli
        """
        if pauli_type == None:
            assert(2 == 3)

        U = None
        if pauli_type == 'Z':
            U = np.matrix([[1., 0.], [0., -1.]])
        elif pauli_type == 'Y':
            U = np.matrix([[0., (0.0 - 1.0j)], [(0.0 + 1.0j), 0.]])
        elif pauli_type == 'X':
            U = np.matrix([[0.0, 1.0], [1.0, 0.0]])
        else:
            assert(2 == 3)

        def tmp(coef):
            return expm((0.0 - 1.0j) * (coef / 2.0) * U)

        return block_diag(tmp(-3.0),
                          tmp(-2.0),
                          tmp(-1.0),
                          tmp(0.0),
                          tmp(1.0),
                          tmp(2.0),
                          tmp(3.0),
                          np.identity(2))

    def train(self):
        """
        Main Function for our learning algorithm, showcasing MoMGrad on 
        (very small) Quanum data.

        We use our oscillator as parameters to learn the pauli coeffcients
        in the exponent of our true unitary We do this by specifying the 
        true unitary we want to learn in the form 

        e^(-i/2 * theta1 * X) * e^(-i/2 * theta2 * Y) * e^(-i/2 * theta3 * Z) = U

        (can also be seen as Rx, Ry and Rz parameters)

        our algorithm will then proceed to draw random 1qubit states
        (basically just sampling on the surface of a sphere). Next we will send
        these random states through this U, enact a phase corresponding to our
        cost function and then "uncompute" U. What remains after this 
        uncompute is performed is the direction and value with which we should
        update our oscillators. This lets us proceed with standard gradient
        descent on our oscillators to carry out learning.

        """

        # Here we draw 400 'samples' for our learning algorithm
        # Note that technically we can create circuits to do sampling in a
        # more "physically realizeable" fashion , however this is simpler to code
        # and in essence is the same

        data_samples = []
        for i in range(self.n_samples):
            data_samples.append(list(self.random_point_sample()))


        # Begin Learning!
        for KKKK in range(4):
            delta = 1
            osc0_c = 0.
            osc1_c = 0.
            osc2_c = 0.

            # Loop over our samples
            for datum in data_samples:

                # Get our random samples and apply our unitary to it
                _, prep_U, initial_rand_state, unitary_applied_to_rand = datum

                # Initialize our oscillators to the learned mu values so far.
                prep_prog = pq.Program().defgate(
                    'prep0',
                    self.make_oscillator_gate(
                        self.prep_oscillator(self.oscialltor0_stats[0],
                                             self.oscialltor0_stats[1])))

                prep_prog += pq.Program().defgate(
                    'prep1',
                    self.make_oscillator_gate(
                        self.prep_oscillator(self.oscialltor1_stats[0],
                                             self.oscialltor1_stats[1])))

                prep_prog += pq.Program().defgate(
                    'prep2',
                    self.make_oscillator_gate(
                        self.prep_oscillator(self.oscialltor2_stats[0],
                                             self.oscialltor2_stats[1])))

                # prep_prog += pq.Program().defgate('prep_in', prep_U)

                # Phase kick the update for our oscillator mu's
                middle_prog = pq.Program().defgate(
                    'bottom_phase',
                    self.make_bottom_phase_gate(-1.0 * self.learning_rate,
                                                unitary_applied_to_rand))

                # Paramterized Oscillator pauli gates
                para_prog = pq.Program().defgate('prep_in', prep_U)
                para_prog += pq.Program().defgate(
                    'oscialltor_X',
                    self.make_controlled_pauli('X'))
                para_prog += pq.Program().defgate(
                    'oscialltor_Y',
                    self.make_controlled_pauli('Y'))
                para_prog += pq.Program().defgate(
                    'oscialltor_Z',
                    self.make_controlled_pauli('Z'))

                # State prep for qubits and oscillators
                # args = ('prep_in', ) + tuple(a for a in self.datum_qubits)
                # prep_prog.inst(args)

                args = ('prep0', ) + tuple(a for a in self.oscialltor0)
                prep_prog.inst(args)

                args = ('prep1', ) + tuple(a for a in self.oscialltor1)
                prep_prog.inst(args)

                args = ('prep2', ) + tuple(a for a in self.oscialltor2)
                prep_prog.inst(args)

                # Program for Oscillator controlled paulis
                args = ('prep_in', ) + \
                    tuple(a for a in self.datum_qubits)
                para_prog.inst(args)

                args = ('oscialltor_X', ) + \
                    tuple(a for a in self.oscialltor0) + \
                    tuple(a for a in self.datum_qubits)
                para_prog.inst(args)

                args = ('oscialltor_Y', ) + \
                    tuple(a for a in self.oscialltor1) + \
                    tuple(a for a in self.datum_qubits)
                para_prog.inst(args)

                args = ('oscialltor_Z', ) + \
                    tuple(a for a in self.oscialltor2) + \
                    tuple(a for a in self.datum_qubits)
                para_prog.inst(args)

                # Program for phase kicking
                args = ('bottom_phase', ) + tuple(a for a in self.datum_qubits)
                middle_prog.inst(args)

                # Program for measuring our oscillators.
                measure_prog = pq.Program().defgate('qftinv',
                                                    self.make_final_qft_inv())
                args = ('qftinv', ) + tuple(a for a in self.oscialltor0)
                measure_prog.inst(args)

                args = ('qftinv', ) + tuple(a for a in self.oscialltor1)
                measure_prog.inst(args)

                args = ('qftinv', ) + tuple(a for a in self.oscialltor2)
                measure_prog.inst(args)

                final_prog = prep_prog + para_prog +\
                    middle_prog + para_prog.dagger() + \
                    measure_prog

                # Keep logs and do update of our oscillators as parameters
                # Note we are effectively minibatching wiht batch size 20 here.
                # 21 - 1 = 20
                if delta % 21 == 0:
                    delta = 1

                    self.oscialltor0_stats[0] += osc0_c / 20.
                    self.oscialltor0_stats[0] = min(
                        self.oscialltor0_stats[0], 3.0)
                    self.oscialltor0_stats[0] = max(
                        self.oscialltor0_stats[0], -3.0)

                    self.oscialltor1_stats[0] += osc1_c / 20.
                    self.oscialltor1_stats[0] = min(
                        self.oscialltor1_stats[0], 3.0)
                    self.oscialltor1_stats[0] = max(
                        self.oscialltor1_stats[0], -3.0)

                    self.oscialltor2_stats[0] += osc2_c / 20.
                    self.oscialltor2_stats[0] = min(
                        self.oscialltor2_stats[0], 3.0)
                    self.oscialltor2_stats[0] = max(
                        self.oscialltor2_stats[0], -3.0)

                    self.param_values.append([self.oscialltor0_stats[0],
                                              self.oscialltor1_stats[0],
                                              self.oscialltor2_stats[0]])

                    self.fid_values.append(self.fidelity(
                        self.oscialltor0_stats[0],
                        self.oscialltor1_stats[0],
                        self.oscialltor2_stats[0]))

                    osc0_c = 0.
                    osc1_c = 0.
                    osc2_c = 0.

                    quantum_soln = self.apply_para_gate(
                        self.oscialltor0_stats[0],
                        self.oscialltor1_stats[0],
                        self.oscialltor2_stats[0])
                    orig_soln = self.apply_para_gate(
                        self.THETA1, self.THETA2, self.THETA3)

                    # Console logging
                    print 'Oscillator info:'
                    print self.oscialltor0_stats
                    print self.oscialltor1_stats
                    print self.oscialltor2_stats
                    print '-'*80
                    print 'True Unitary:'
                    print orig_soln

                    print 'Unitary learned so far:'
                    print quantum_soln

                else:
                    delta += 1
                    osc0_c += self.vqe_inst.expectation(
                        final_prog, self.get_measure_ham(self.oscialltor0),
                        None,
                        self.qvm)
                    osc1_c += self.vqe_inst.expectation(
                        final_prog, self.get_measure_ham(self.oscialltor1),
                        None,
                        self.qvm)
                    osc2_c += self.vqe_inst.expectation(
                        final_prog, self.get_measure_ham(self.oscialltor2),
                        None,
                        self.qvm)

            random.shuffle(data_samples)

            self.oscialltor0_stats[1] -= 0.1
            self.oscialltor1_stats[1] -= 0.1
            self.oscialltor2_stats[1] -= 0.1

        print 'Done!'


# """ Simple Use Case example """

# qvm = api.SyncConnection('http://127.0.0.1:5000')

# arbitrary_1qubit = MoMGrad1QB(qvm, -1./3., 2., -1.)

# arbitrary_1qubit.train()

# print arbitrary_1qubit.fid_values
