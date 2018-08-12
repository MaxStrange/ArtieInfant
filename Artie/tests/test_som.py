"""
This module contains all the tests for the Self-Organizing Map
"""
import numpy as np
import os
import sys
import unittest
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, path)
import internals.som.som as som # pylint: disable=locally-disabled, import-error

class TestSequence(unittest.TestCase):
    def setUp(self):
        pass

    def test_make_som(self):
        """
        Test creating a SOM object.
        """
        nn = som.SelfOrganizingMap(shape=(5, 4, 2), weights=np.zeros((5, 40)))
        self.assertEqual((5, 4, 2), nn.shape)

        nn = som.SelfOrganizingMap(shape=(1, 1, 1), weights=np.ones((1, 1)))
        self.assertEqual((1, 1, 1), nn.shape)

        self.assertRaises(ValueError, som.SelfOrganizingMap, (1, 4), (1, 4))
        self.assertRaises(ValueError, som.SelfOrganizingMap, (1, 4, 0), (5, 5))
        self.assertRaises(ValueError, som.SelfOrganizingMap, (-1, 4, 1), (2, 3))

        shape = (1, 3, 4)
        weights = np.ones(shape=(2, np.prod(shape)))
        nn = som.SelfOrganizingMap(shape=shape, weights=weights)
        self.assertTrue(np.all(nn.weights == weights))

        self.assertRaises(ValueError, som.SelfOrganizingMap, (5, 4, 1), np.ones(shape=(4, 21)))

        nn = som.SelfOrganizingMap(shape=(5, 4, 1), weights=np.ones(shape=(4, 20)))
        self.assertTrue(np.all(nn.weights == np.ones(shape=(5, 4, 1))))

    def test_activation_shape(self):
        """
        Tests the shape of the output of the map's activation.
        """
        # Test the shape of the output
        shape = (5, 4, 3)
        weights = np.random.sample(size=(3, np.prod(shape)))
        nn = som.SelfOrganizingMap(shape=shape, weights=weights)
        nninput = np.random.sample(size=shape)
        nnoutput = nn.activate(nninput)
        self.assertEqual(nnoutput.shape[0], weights.shape[0])

        # Another test
        shape = (1, 1, 1)
        weights = np.random.sample((4, 1))
        nn = som.SelfOrganizingMap(shape=shape, weights=weights)
        nninput = np.random.sample(size=shape)
        nnoutput = nn.activate(nninput)
        self.assertEqual(nnoutput.shape[0], weights.shape[0])

    def test_output_simple(self):
        """
        Tests activating the map for output.
        """
        inshape = (2, 3, 1)
        outshape = 2
        weights = np.ones(shape=(outshape, np.prod(inshape)))
        nn = som.SelfOrganizingMap(shape=inshape, weights=weights)
        # Input:
        # 1.0  0.4  1.0
        # 1.0  1.0  1.0
        nninput = np.ones(inshape)
        nninput[0, 1, 0] = 0.4
        # Weights:
        # 1.0  1.0  1.0  1.0  1.0  1.0
        # 1.0  1.0  1.0  1.0  1.0  1.0

        # Input after normalization:
        # 1/5.4 0.4/5.4 1/5.4
        # 1/5.4 1/5.4   1/5.4

        # Therefore, expected activation:
        # 0.185  0.074  0.185  0.185  0.185  0.185
        # 0.185  0.074  0.185  0.185  0.185  0.185

        val = np.sum(nninput / np.sum(nninput))
        expected_output = val * np.ones((2, 1))
        # Get the output
        nnoutput = nn.activate(nninput)
        self.assertTrue(np.allclose(nnoutput, expected_output))

    def test_output_weights_zeros(self):
        """
        Tests that outputs are all zeros, given that the weights
        are all zeros.
        """
        inshape = (5, 5, 5)
        outshape = 7

        # Zero the weights in the SOM
        weights = np.zeros(shape=(outshape, np.prod(inshape)))

        # Make the SOM
        nn = som.SelfOrganizingMap(shape=inshape, weights=weights)

        # Make a random input
        nninput = np.random.sample(inshape)

        # Expect only zeros on the output, since all the weights are zeros
        expected = np.zeros(outshape)

        # Activate the map and test that the output matches expected (all zeros)
        nnoutput = nn.activate(nninput)
        self.assertTrue(np.all(nnoutput == expected))

        # Now do the same, but with lateral inhibition
        nnoutput = nn.activate_with_lateral_inhibition(nninput)
        self.assertTrue(np.all(nnoutput == expected))

    def test_maximum_output(self):
        """
        Tests that output is never more than expected.
        """
        inshape = (10, 10, 2)
        outshape = 1000

        # Make some random weights, all between 0 and 1
        weights = np.random.sample(size=(outshape, np.prod(inshape)))

        # Make the SOM
        nn = som.SelfOrganizingMap(inshape, weights=weights)

        # Create an input of random values between 0 and 1
        nninput = np.random.sample(inshape)

        # Get an output from the SOM
        nnoutput = nn.activate(nninput)

        # We expect that none of the outputs are greater than np.prod(inshape)
        maxval = np.prod(inshape)
        all_values_lt_max = len(np.where(nnoutput >= maxval)[0]) == 0
        self.assertTrue(all_values_lt_max)

    def test_activate_w_lat_inh_shape(self):
        """
        Test activation with lateral inhibition. Specifically, test the shape
        of the output.
        """
        inshape = (5, 4, 3)
        outshape = 3

        # Set up random weights
        weights = np.random.sample(size=(outshape, np.prod(inshape)))

        # Create SOM
        nn = som.SelfOrganizingMap(shape=inshape, weights=weights)

        # Make a random input
        nninput = np.random.sample(size=inshape)

        # Get the SOM's output. Should be outshape
        nnoutput = nn.activate_with_lateral_inhibition(nninput)
        self.assertEqual(nnoutput.shape[0], weights.shape[0])
        self.assertEqual(nnoutput.shape[0], outshape)

    def test_activate_w_lat_inh_simple_shape(self):
        """
        Test lateral inhibition activation with very simple shape.
        """
        inshape = (1, 1, 1)
        outshape = 3

        # Set up random weights
        weights = np.random.sample(size=(outshape, np.prod(inshape)))

        # Create SOM
        nn = som.SelfOrganizingMap(shape=inshape, weights=weights)

        # Make a random input
        nninput = np.random.sample(size=inshape)

        # Get the SOM's output. Should be outshape
        nnoutput = nn.activate_with_lateral_inhibition(nninput)
        self.assertEqual(nnoutput.shape[0], weights.shape[0])
        self.assertEqual(nnoutput.shape[0], outshape)

    def test_activate_w_lat_inh(self):
        """
        Test activation with lateral inhibition.
        """
        # Test that the output of the map is what I expect, given a known input and weights
        inshape = (2, 3, 1)
        outshape = 2

        # Weights of the map will be all 1.0
        weights = np.ones(shape=(outshape, np.prod(inshape)))

        # Create SOM
        nn = som.SelfOrganizingMap(shape=inshape, weights=weights)

        # Input is:
        # [ 0.2 0.4 0.3 ]
        # [ 0.1 0.3 0.0 ]
        nninput = np.zeros(inshape)
        nninput[0, 1, 0] = 0.4
        nninput[0, 0, 0] = 0.2
        nninput[0, 2, 0] = 0.3
        nninput[1, 1, 0] = 0.3
        nninput[1, 0, 0] = 0.1

        # Expected internal tmp (after lateral inhibition) is:
        # [ 0.2 0.4 0.3 ]
        # [ 0.0 0.3 0.0 ]

        # Expected internal tmp after normalization is:
        # [ 0.2/1.2 0.4/1.2 0.3/1.2 ]
        # [ 0.0/1.2 0.3/1.2 0.0/1.2 ]
        # =
        # [ 0.166667  0.33333  0.25 ]
        # [ 0.0       0.25     0.0

        # Expected output is:
        # SUM([ 0.166667 0.333333 0.25 0.0 0.25 0.0 ]) = [1.0
        # SUM([ 0.166667 0.333333 0.25 0.0 0.25 0.0 ])    1.0]
        expected = np.zeros(shape=weights.shape)
        expected[:, 0] = 0.1666666666666666667
        expected[:, 1] = 0.3333333333333333333
        expected[:, 2] = 0.25
        expected[:, 4] = 0.25
        expected_output = np.sum(expected, axis=1)

        # Get the output and test
        nnoutput = nn.activate_with_lateral_inhibition(nninput)
        self.assertTrue(np.allclose(nnoutput, expected_output))

    def test_lateral_inh_with_non_unity_weights(self):
        """
        Test the lateral inhibition, but have weights that aren't all 1s.
        """
        inshape = (2, 3, 1)
        outshape = 2

        # Weights:
        # 1.0  0.5  1.2  0.25  0.0  4.0
        # 0.9  0.75 1.5  1.10  0.1  0.1
        weights = np.array([[1.0, 0.5, 1.2, 0.25, 0.0, 4.0],
                            [0.9, 0.75,1.5, 1.10, 0.1, 0.1]])

        # Create the SOM
        nn = som.SelfOrganizingMap(shape=inshape, weights=weights)

        # Input is:
        # [ 0.2  0.2  0.8 ]
        # [ 0.1  0.7  0.3 ]
        nninput = np.array([[0.2, 0.2, 0.8],
                            [0.1, 0.7, 0.3]]).reshape(inshape)

        # Expected internal tmp (after lateral inhibition) is:
        # [ 0.0  0.2  0.8 ]
        # [ 0.0  0.0  0.3 ]

        # Expected internal tmp (after normalization) is:
        # [ 0.0  0.2/1.3  0.8/1.3 ]
        # [ 0.0  0.0      0.3/1.3 ]
        # =
        # [ 0.0  0.1538461538461 0.6153846 ]
        # [ 0.0  0.0             0.2307692 ]

        # Expected output is:
        # SUM([ (0.0 * 1.0)  (0.1538 * 0.5)  (0.6153 * 1.2)  (0.0 * 0.25)  (0.0 * 0.0)  (0.2307 * 4.0) ])
        # SUM([ (0.0 * 0.9)  (0.1538 * 0.75) (0.6153 * 1.5)  (0.0 * 1.10)  (0.0 * 0.1)  (0.2307 * 0.1) ])
        # =
        # [2.26, 1.38]
        output = nn.activate_with_lateral_inhibition(nninput)
        self.assertAlmostEqual(output[0, 0], (0.5 * 0.2/1.3) + (1.2 * 0.8/1.3) + (4.0 * 0.3/1.3))
        self.assertAlmostEqual(output[1, 0], (0.75 * 0.2/1.3) + (1.5 * 0.8/1.3) + (0.1 * 0.3/1.3))

    def test_with_non_unity_weights(self):
        """
        Test the normal activation, but with realistic weights.
        """
        inshape = (2, 3, 1)
        outshape = 2

        # Weights:
        # 1.0  0.5  1.2  0.25  0.0  4.0
        # 0.9  0.75 1.5  1.10  0.1  0.1
        weights = np.array([[1.0, 0.5, 1.2, 0.25, 0.0, 4.0],
                            [0.9, 0.75,1.5, 1.10, 0.1, 0.1]])

        # Create the SOM
        nn = som.SelfOrganizingMap(shape=inshape, weights=weights)

        # Input is:
        # [ 0.2  0.2  0.8 ]
        # [ 0.1  0.7  0.3 ]
        nninput = np.array([[0.2, 0.2, 0.8],
                            [0.1, 0.7, 0.3]]).reshape(inshape)

        # Input after normalization:
        # [ 0.087 0.087 0.348 ]
        # [ 0.044 0.304 0.130 ]

        output = nn.activate(nninput)
        sm = np.sum(nninput)
        self.assertAlmostEqual(output[0, 0], (1.0 * 0.2/sm) + (0.50 * 0.2/sm) + (1.2 * 0.8/sm) + (0.25 * 0.1/sm) + (0.0 * 0.7/sm) + (4.0 * 0.3/sm))
        self.assertAlmostEqual(output[1, 0], (0.9 * 0.2/sm) + (0.75 * 0.2/sm) + (1.5 * 0.8/sm) + (1.10 * 0.1/sm) + (0.1 * 0.7/sm) + (0.1 * 0.3/sm))

    def test_reinforcement(self):
        """
        Test reinforcing the network.
        """
        # Expected use case: reinforce somewhere in the middle, normal sized radius and lr
        # Network:
        # n0  n1  n2  n3
        # n4  n5  n6  n7
        # n8  n9  n10 n11
        # If we keep reinforcing n5 with a radius of 1, we should see n5, n1, n4, n6, and n9 all
        # increase their weights.

        # Set up the network
        inshape = (3, 4, 1)
        outshape = 3
        weights = np.random.sample(size=(outshape, np.prod(inshape)))
        nn = som.SelfOrganizingMap(shape=inshape, weights=weights)

        # Create an input that has maximum value at n5, otherwise, random
        nninput = np.random.sample(size=inshape)
        nninput[1, 1, 0] = 1.0
        nn.activate_with_lateral_inhibition(nninput)

        # Get the weights before reinforcement
        # We are interested in the following nodes:
        ns_of_interest = [1, 4, 5, 6, 9]
        other_ns = [i for i in range(np.prod(inshape)) if i not in ns_of_interest]

        # Our weights of interest are therefore each column of weights for each of the indices
        weights_of_interest_before = [np.copy(nn.weights[:, i]) for i in ns_of_interest]
        other_weights_before = [np.copy(nn.weights[:, i]) for i in other_ns]

        # Set up some simple args for the reinforcement
        m = [1 for _ in range(outshape)]
        radius = 1.0
        lr = 1.0

        # Reinforce
        nn.reinforce(m, radius, lr)

        # Get the weights after
        weights_of_interest_after = [nn.weights[:, i] for i in ns_of_interest]
        other_weights_after = [nn.weights[:, i] for i in other_ns]

        # Check that the ones we expected to change did, and the ones we did not expect to change didn't
        for i, (before, after) in enumerate(zip(weights_of_interest_before, weights_of_interest_after)):
            node = ns_of_interest[i]
            self.assertTrue(np.all(after > before), "Node {} did not increase. Before {}, after {}".format(node, before, after))
        for i, (before, after) in enumerate(zip(other_weights_before, other_weights_after)):
            node = other_ns[i]
            for b, a in zip(before, after):
                self.assertAlmostEqual(b, a, "Node {} changed value. Before {}, after {}".format(node, before, after))

