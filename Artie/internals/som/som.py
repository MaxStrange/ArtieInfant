"""
Self-Organizing Map

This module contains code for creating and running a self-organizing map.
"""
import itertools
import math
import numpy as np
import random

class SelfOrganizingMap:
    """
    The main class for this module. This is the interface to use for creating and using
    a Self Organizing Map.

    The SOM is a 3D volume of neurons connected to a 1D vector of output nodes in a dense
    fashion. The connections come with an algorithm for changing them over time.

    - The inputs are shaped m x n x o.
    - The outputs are shaped p.
    - The weight matrix (the connections between input and output) are shaped p x mno.

    The typical use of this class is to create an instance with a particular input shape
    and an initial weight matrix, and then to activate with lateral inhibition to get
    an output vector (where each item in the vector has some amount of activation from
    each node on the input). Then the reinforcement algorithm increases the weights
    of each node within some euclidean distance of the most activated node.

    The point of the SOM is twofold; after a reinforcement regimen:

    1) Similar inputs should produce similar outputs
    2) Outputs will be constrained to those values that were reinforced, regardless
       of inputs.
    """
    def __init__(self, shape, weights):
        """
        Constructor.

        Note that the way this SOM works is different from typical ones, which are
        used for dimensionality reduction. This SOM is used instead for shaping
        a map that can be used as a function that takes an input of shape `shape`
        and outputs a value of shape weights.shape[0].

        The typical usage is to set up the SOM by giving it a map shape and
        an output weight shape, then iterating through trials like this:

        nn = SelfOrganizingMap(shape, weights)
        for e in range(epochs):
            inp = x # somehow generate an input
            output = nn.activate(x)  # x has shape nn.shape
            nn.reinforce(m=some_function_of_output, radius=some_threshold, lr=some_learning_rate)  # Update the weights

        Note that `weights` is of shape (whatever, n_nodes_in_som), so that if we have
        a SOM that looks like this:

        n0 n1 n2
        n3 n4 n5

        We might have a weights matrix that looks like this:

        w_n0_o0 w_n1_o0 w_n2_o0
        w_n0_o1 w_n1_o1 w_n2_o1
        w_n0_o2 w_n1_o2 w_n2_o2
        w_n0_o3 w_n1_o3 w_n2_o3

        (where w_nx_oy is the weight of the connection from node x to output y). The number of
        columns must equal the number of nodes in the SOM.

        :param shape:   The shape of the Map. Must be exactly three dimensions, and
                        each axis must be positive.
        :param weights: The initial weight matrix. The weight matrix represents the strength
                        of the connection between each neuron in the SOM and whatever output
                        you will be feeding the activations into. In Artie, this is at least
                        the Praat synthesizer module, but I've written this class to allow
                        for anything, as long as it is fully connected (though if you want
                        some sparseness, you can just supply zeros in whatever slot you want).
                        The shape must be (whatever, n_nodes_in_som).
        """
        # Sanitize the input args
        if len(shape) != 3:
            raise ValueError("Need a 3D shape, got shape {}".format(shape))
        elif len(np.where(np.array(shape) <= 0)[0]) > 0:
            raise ValueError("Need a 3D shape with axes greater than or equal to 1, but got shape {}".format(shape))

        if len(weights.shape) != 2:
            raise ValueError("Need a 2D weights matrix, got matrix of shape {}".format(weights.shape))
        elif weights.shape[1] != np.prod(shape):
            raise ValueError("Need a 2D weights matrix of shape (whatever, {}), but got shape {}".format(np.prod(shape), weights.shape))

        self.weights = weights
        self.shape = shape
        self.latest_activation = None
        self.latest_nninput = None

    def activate(self, nninput):
        """
        Takes a `nninput`, which is a matrix of the same shape as self.shape,
        and applies it to the weights in the SOM so that each node's input value
        is multiplied by each of its weights to produce the activation matrix.

        :param nninput: The input to the network. Must be of same shape as self.shape.
        :returns:       The result of multiplying each node's value by its weights, then summing
                        across the weights matrix's rows - so the result is a vector of shape p,
                        given a nninput of shape m x n x o and a weight matrix of shape p x mno.
                        Vector element p_i is equal to SUM((a*b)[i, :]) where 'a' is reshaped nninput
                        and b is reshaped weights.
        """
        if nninput.shape != self.shape:
            raise ValueError("Shape of nninput must be {} but got {}".format(self.shape, nninput.shape))
        self.latest_nninput = np.copy(nninput)

        a = np.reshape(nninput, (1, int(np.prod(nninput.shape)), 1))
        a = a / np.sum(a)  # normalize the activations
        b = self.weights[:, :, np.newaxis]
        activation = np.reshape(a * b, self.weights.shape)  # broadcast into correct shape by concatenating vertically

        self.latest_activation = np.copy(activation)
        result = np.sum(activation, axis=1)
        return np.reshape(result, (len(result), 1))

    def activate_with_lateral_inhibition(self, nninput):
        """
        Same as `activate()`, but zeros the inputs of all but the most activated neuron
        and its immediate (non-diagonal) neighbors in `nninput`.

        :param nninput: The input to the network. Must be of same shape as self.shape.
        :returns:       The result of multiplying each node's value by its weights,
                        but zeroed on all nodes except the most activated and its
                        immediate orthogonal neighbors.
        """
        nninput = np.copy(nninput)

        # Get the x, y, and z index of the single most activated node
        indices_max = np.unravel_index(np.argmax(nninput), nninput.shape)

        # Find all the nodes that tie this one for most activated
        x, y, z = indices_max
        maxval  = nninput[x, y, z]
        tol     = 1E-6
        tiemask = (nninput == maxval) & (nninput <= (maxval + tol)) & (nninput >= (maxval - tol))
        maxidxs = np.where(tiemask)
        maxxs   = [x for x in maxidxs[0]]
        maxys   = [y for y in maxidxs[1]]
        maxzs   = [z for z in maxidxs[2]]

        # Get all the nodes that neighbor these nodes orthogonally
        neighbors_and_max = []
        for x, y, z in zip(maxxs, maxys, maxzs):
            idxs = [
                (x, y - 1),
                (x, y + 1),
                (x - 1, y),
                (x + 1, y),
                (x, y, z - 1),
                (x, y, z + 1)
            ]

            neighbors_and_max.append((x, y, z))
            for xyz in idxs:
                oob = False
                for i, x_y_or_z in enumerate(xyz):
                    if x_y_or_z < 0 or x_y_or_z >= nninput.shape[i]:
                        oob = True
                if not oob:
                    neighbors_and_max.append(xyz)

        # Zero other activations
        activations_to_save = [nninput[idxs] for idxs in neighbors_and_max]
        nninput = np.zeros_like(nninput)
        for i, act in zip(neighbors_and_max, activations_to_save):
            nninput[i] = act

        activation  = self.activate(nninput)

        return activation

    def reinforce(self, m, radius, lr=0.8, nninput=None):
        """
        Modifies the weights of the most activated neuron and its
        neighbors within the euclidean distance less than or equal to `radius`.

        Formally,

        W_p' = { W_p + lr(m - W_p) if sqrt( (X_p - X_q)^2 + (Y_p - Y_q)^2 + (Z_p - Z_q)^2 ) <= radius
               { W_p

        Algorithm is this:
        - Find which neuron is the most active in nninput.
        - Get all nodes which are within radius of that node in nninput.
        - Calculate which columns these nodes are in the weight matrix.
        - Increase the columns' values for each row by lr * m - previous value.

        :param m:       A vector of same length as the output of this SOM.
                        This parameter modulates the change in the weights of reinforced neurons.
                        Specifically, if lr=1.0, a reinforcement event changes the weights of the
                        most activated neurons to be equal to this value.
        :param radius:  The radius around the most active neuron within which the reinforcement will take place.
        :param lr:      Learning rate. Must be in interval (0, 1.0].
        :param nninput: The SOM input to use to determine which node was the most active. If None,
                        will use the latest one.
        :returns:       None. Modifies the internal weights of the network.
        """
        if nninput is None:
            nninput = self.latest_nninput

        if len(m) != self.weights.shape[0]:
            raise ValueError("m is length {} but should be of length {}".format(len(m), self.weights.shape[0]))
        if nninput is None:
            raise ValueError("nninput must be specified if the SOM does not have a latest_nninput")
        if nninput.shape != self.shape:
            raise ValueError("nninput is of shape {} but should be {}".format(nninput.shape, self.shape))
        if lr <= 0.0:
            raise ValueError("lr is {}, but must be 0 < lr <= 1.0".format(lr))
        if lr > 1.0:
            raise ValueError("lr is {}, but must be 0 < lr <= 1.0".format(lr))

        m = np.array(m)

        # Get the x, y, and z index of the maximum value in the input
        x0, y0, z0 = np.unravel_index(np.argmax(nninput), nninput.shape)

        # Get the unrolled index - the index of the node's column in the weights matrix
        npercol = nninput.shape[0]
        nperrow = nninput.shape[1]
        nperlayer = nperrow * npercol
        nodeidx = (nperlayer * z0) + (nperrow * x0) + y0

        ns = []
        # Now collect each node that is within radius of the input
        # (this will also gather up the target node itself, since its dist to itself is 0)
        for x1 in range(nninput.shape[0]):
            for y1 in range(nninput.shape[1]):
                for z1 in range(nninput.shape[2]):
                    dist = math.sqrt((x0 - x1)**2 + (y0 - y1)**2 + (z0 - z1)**2)
                    if dist <= radius:
                        idx = (nperlayer * z1) + (nperrow * x1) + y1
                        ns.append(idx)

        for n in ns:
            self.weights[:, n] += lr * (m - self.weights[:, n])

def _get_avg_output(nn, nactivations, inputdim, outputdim, nninput=None):
    outputs = np.zeros(shape=(nactivations, outputdim))
    for i in range(nactivations):
        if nninput is None:
            nninput = np.random.sample(size=(inputdim, inputdim, inputdim))
        outvect = nn.activate_with_lateral_inhibition(nninput)
        outvect = np.reshape(outvect, (outputdim,))
        outputs[i, :] = outvect[:]
    avg = np.mean(outputs, axis=0)
    return avg

def _reinforce(nn, tol, inputdim, target, radius, lr, nninput=None):
    nactivations_needed = 0
    while True:
        nactivations_needed += 1
        if nninput is None:
            nninput = np.random.sample(size=(inputdim, inputdim, inputdim))
        outvect = nn.activate_with_lateral_inhibition(nninput)
        if np.allclose(outvect, target, rtol=0.0, atol=tol):
            nn.reinforce(target, radius, lr)
            return nactivations_needed
        if nactivations_needed > 50:
            return 50

def _plot_results(average_values, lengths, average_weights, target):
    import matplotlib.pyplot as plt
    plt.subplot(3, 1, 1)
    plt.title("Average Output Over Time")
    plt.ylabel("Average Output [0]")
    plt.plot(average_values)
    plt.plot([target[0] for _ in average_values], color='orange', linestyle='--')

    plt.subplot(3, 1, 2)
    plt.title("Number of Activations Needed to Get Result")
    plt.ylabel("Number of Activations Required")
    plt.plot(lengths)
    plt.plot([1.0 for _ in lengths], color='orange', linestyle='--')

    plt.subplot(3, 1, 3)
    plt.title("Average Value of Weights in SOM")
    plt.ylabel("Average Weight Value")
    plt.plot(average_weights)

    plt.show()

def self_test(inputdim=10, outputdim=1, target=0.4, tolstart=0.5, tolend=0.1, nepochs=10, radius=1.0, lr=0.8):
    """
    This method tests the functionality of the SOM.

    Specifically, it does the following:

    1. Initialize a cube SOM of inputdim ** 3, outputdim=outputdim.
    2. Activate the SOM with random uniform activations drawn from [0, 1)
       and average the output vector over 100 activations.
    3. Plot this value.
    4. Activate with random input until the output is within tolerance of target.
    5. Reinforce.
    6. Activate 100 times again and plot.
    7. Keep doing this for nepochs.

    We then plot two graphs. The top one is the average value of output vector item 0
    over time. This value should converge on target.
    The bottom plot is the number of activations needed to attain an output value
    that was within tolerance of target, over time. This value should start high
    and converge on 1 over time, meaning that the number of activations to get
    the target should eventually become just 1.

    `target` must be either a scalar, in which case the target will be a vector
    of that value repeated outputdim times, or a vector, in which case it must
    be of length outputdim.
    """
    from tqdm import tqdm

    if inputdim <= 0:
        raise ValueError("inputdim is {} but must be greater than 0".format(inputdim))
    if outputdim <= 0:
        raise ValueError("outputdim is {} but must be greater than 0".format(outputdim))
    if tolstart <= 0:
        raise ValueError("negative and zero tolerances are meaningless")
    if tolend <= 0:
        raise ValueError("negative and zero tolerances are meaningless")
    if nepochs <= 0:
        raise ValueError("nepochs is {} but must be greater than 0".format(nepochs))
    try:
        _ = target[0]
        target = np.array(target)
        assert target.shape[0] == outputdim, "target must be scalar or vector which is same shape as outputdim"
    except TypeError:
        target = target * np.ones(outputdim)
    tolerances = np.linspace(tolstart, tolend, num=nepochs)

    ws = np.random.sample(size=(outputdim, inputdim ** 3))
    nn = SelfOrganizingMap(shape=(inputdim, inputdim, inputdim), weights=ws)

    nactivations = 100

    # Get the initial average output
    average_weights = []
    average_values = []
    avg = _get_avg_output(nn, nactivations, inputdim, outputdim)
    average_values.append(avg)
    average_weights.append(np.copy(np.mean(nn.weights)))

    # Now reinforce the network, checking how long it takes to do so,
    # then get average output again, and repeat for nepochs
    lengths = []
    for i in tqdm(range(nepochs)):
        tol = tolerances[i]
        nacts = _reinforce(nn, tol, inputdim, target, radius, lr)
        avg = _get_avg_output(nn, nactivations, inputdim, outputdim)
        average_weights.append(np.copy(np.mean(nn.weights)))

        average_values.append(avg)
        lengths.append(nacts)

    _plot_results(average_values, lengths, average_weights, target)

def self_test_two(target_a=0.1, target_b=0.9):
    """
    Inputs two different input masks, A + G and B + G, where G is
    a gaussian noise matrix; chooses which one randomly with
    50/50 probability.

    Reinforces for A => target_a and B => target_b.
    """
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    nninshape = (10, 10, 1)
    outshape  = 1
    weights   = np.random.sample(size=(outshape, np.prod(nninshape)))
    nn        = SelfOrganizingMap(nninshape, weights)

    # template A - a big plus sign
    templatea = np.zeros(shape=nninshape)
    half_r, half_c = int(round(nninshape[0]/2.0)), int(round(nninshape[1]/2.0))
    templatea[:, half_c, :] = 1.0
    templatea[half_r, :, :] = 1.0

    templateb = np.zeros(shape=nninshape)
    maxcol = 4
    assert maxcol >= 2
    assert nninshape[1] > maxcol
    templateb[:, maxcol - 2, :] = 0.5
    templateb[:, maxcol - 1, :] = 0.4
    templateb[:, maxcol, :] = 0.6

    # For some number of trials, draw a random template (A or B), then mutate it slightly and activate.
    # If we get close to A's target for A or B's target for B, reinforce. Otherwise ignore.
    # Afterwards, get the average values for each input.
    # Do this for some number of epochs.
    nepochs = 100
    ntrials = 500
    templates = (templatea, templateb)
    targets   = ([target_a], [target_b])
    averagesa = []
    averagesb = []
    sigmas    = [0.5, 0.5]
    mus       = [1.0, 0.5]
    for e in tqdm(range(nepochs)):
        for t in range(ntrials):
            # Choose which template, then get its noise
            idx      = random.choice((0, 1))
            template = templates[idx]
            mu       = mus[idx]
            gaussian_noise = np.random.normal(mu, sigmas[idx], size=nninshape)

            # Create the input - constrain the noisy input to [0, 1]
            nninput  = template + gaussian_noise
            nninput[nninput > 1.0] = 1.0
            nninput[nninput < 0.0] = 0.0

            # Activate the network
            output   = nn.activate_with_lateral_inhibition(nninput)

            # Check if the result is worth reinforcing
            target   = targets[idx]
            if np.allclose(output, target, atol=0.4):
                nn.reinforce(target, radius=1.0, lr=0.8)
                # if we reinforce, anneal the standard deviations a bit
                sigmas[idx] *= 0.8

        # Activate the network a bunch to retrieve averages and log them for plotting
        avga = _get_avg_output(nn, nactivations=5, inputdim=nninshape, outputdim=outshape, nninput=templatea)
        avgb = _get_avg_output(nn, nactivations=5, inputdim=nninshape, outputdim=outshape, nninput=templateb)
        averagesa.append(avga)
        averagesb.append(avgb)

    plt.subplot(2, 1, 1)
    plt.title("Average Output Over Time For Template A")
    plt.ylabel("Average Output [0]")
    plt.plot(averagesa)
    plt.plot([target_a for _ in averagesa], color='orange', linestyle='--')

    plt.subplot(2, 1, 2)
    plt.title("Average Output Over Time For Template B")
    plt.ylabel("Average Output [0]")
    plt.plot(averagesb)
    plt.plot([target_b for _ in averagesb], color='orange', linestyle='--')

    plt.show()

if __name__ == "__main__":
    configdict = {
        'inputdim':   10,
        'outputdim':  1,
        'target':     0.7,
        'tolstart':   0.5,
        'tolend':     0.02,
        'nepochs':    500,
        'radius':     1.0,
        'lr':         0.8,
    }
    #print("Testing SOM with the following values:\n", configdict)
    #self_test(**configdict)

    self_test_two()
