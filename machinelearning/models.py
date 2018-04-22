import numpy as np

import backend
import nn

class Model(object):
    """Base model class for the different applications"""
    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()
            graph.step(self.learning_rate)

class RegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        # self.learning_rate = .05
        # self.learning_rate = .1
        # self.hidden = 50


        self.learning_rate = .05
        self.hidden = 200
        self.m1 = nn.Variable(1, self.hidden)
        self.b1 = nn.Variable(1, self.hidden)
        self.m2 = nn.Variable(self.hidden, 1)
        self.b2 = nn.Variable(1,1)
        # self.graph = nn.Graph([self.m1, self.b1, self.m2, self.b2])
        self.graph = None


    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        # This is our data, where x is a 4x2 matrix and y is a 4x1 matrix
        # Let's construct a simple model to approximate a function from 2D
        # points to numbers, f(x) = x_0 * m_0 + x_1 * m_1 + b
        # Here m and b are variables (trainable parameters):
  
        # We train our network using batch gradient descent on our data
        # for iteration in range(20000):
            # At each iteration, we first calculate a loss that measures how
            # good our network is. The graph keeps track of all operations used
        self.graph = nn.Graph([self.m1, self.m2, self.b1, self.b2])
        input_x = nn.Input(self.graph, x)
        xm1 = nn.MatrixMultiply(self.graph, input_x, self.m1)
        xm1_plus_b = nn.MatrixVectorAdd(self.graph, xm1, self.b1)
        relu = nn.ReLU(self.graph, xm1_plus_b)
        relum2 = nn.MatrixMultiply(self.graph, relu, self.m2)
        func = nn.MatrixVectorAdd(self.graph, relum2, self.b2)
        # input_y = nn.Input(self.graph, y)
        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(self.graph, y)
            loss = nn.SquareLoss(self.graph, func, input_y)
            # self.graph.add(loss)
            return self.graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            return self.graph.get_output(func)

class OddRegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers.

    Unlike RegressionModel, the OddRegressionModel must be structurally
    constrained to represent an odd function, i.e. it must always satisfy the
    property f(x) = -f(-x) at all points during training.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = .1
        self.hidden = 50
        self.m1 = nn.Variable(1, self.hidden)
        self.b1 = nn.Variable(1, self.hidden)
        self.m2 = nn.Variable(self.hidden, 1)
        self.b2 = nn.Variable(1)
      
        self.graph = None





    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
     

        self.graph = nn.Graph([self.m1, self.m2, self.b1, self.b2])
        input_x = nn.Input(self.graph, x)
        matmult = nn.MatrixMultiply(self.graph, input_x, self.m1)
        matadd = nn.MatrixVectorAdd(self.graph, matmult, self.b1)
        relu = nn.ReLU(self.graph, matadd)
        matmult = nn.MatrixMultiply(self.graph, relu, self.m2)
        matadd1 = nn.MatrixVectorAdd(self.graph, matmult, self.b2)

        matrixNeg = nn.Input(self.graph, np.array([[-1.]]))


        matmultNeg = nn.MatrixMultiply(self.graph, input_x, matrixNeg)
        matmult = nn.MatrixMultiply(self.graph, matmultNeg, self.m1)
        matadd = nn.MatrixVectorAdd(self.graph, matmult, self.b1)
        relu = nn.ReLU(self.graph, matadd)
        matmult = nn.MatrixMultiply(self.graph, relu, self.m2)
        matadd2 = nn.MatrixVectorAdd(self.graph, matmult, self.b2)

        negMat = nn.MatrixMultiply(self.graph, matadd2, matrixNeg)
        func = nn.MatrixVectorAdd(self.graph, matadd1, negMat)

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(self.graph, y)
            loss = nn.SquareLoss(self.graph, func, input_y)
            # self.graph.add(loss)
            return self.graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            return self.graph.get_output(func)

class DigitClassificationModel(Model):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = .5
        self.hidden = 400

        self.m1 = nn.Variable(784, self.hidden)
        self.b1 = nn.Variable(1, self.hidden)
        self.m2 = nn.Variable(self.hidden, 10)
        self.b2 = nn.Variable(1, 10)

        self.m3 = nn.Variable(10, self.hidden)
        self.b3 = nn.Variable(1, self.hidden)
        self.m4 = nn.Variable(self.hidden, 10)
        self.b4 = nn.Variable(1, 10)


    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should predict a (batch_size x 10) numpy array of scores,
        where higher scores correspond to greater probability of the image
        belonging to a particular class. You should use `nn.SoftmaxLoss` as your
        training loss.

        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        """
        "*** YOUR CODE HERE ***"
        self.graph = nn.Graph([self.m1, self.m2, self.b1, self.b2, self.m3, self.b3, self.m4, self.b4])
        input_x = nn.Input(self.graph, x)
        xm1 = nn.MatrixMultiply(self.graph, input_x, self.m1)
        xm1_plus_b = nn.MatrixVectorAdd(self.graph, xm1, self.b1)

        relu = nn.ReLU(self.graph, xm1_plus_b)


        relum2 = nn.MatrixMultiply(self.graph, relu, self.m2)
        func = nn.MatrixVectorAdd(self.graph, relum2, self.b2)

        relu = nn.ReLU(self.graph, func)


        xm1 = nn.MatrixMultiply(self.graph, relu, self.m3)
        xm1_plus_b = nn.MatrixVectorAdd(self.graph, xm1, self.b3)

        relu = nn.ReLU(self.graph, xm1_plus_b)
       
        relum2 = nn.MatrixMultiply(self.graph, relu, self.m4)
        func = nn.MatrixVectorAdd(self.graph, relum2, self.b4)

        if y is not None:
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(self.graph, y)
            loss = nn.SoftmaxLoss(self.graph, func, input_y)
            return self.graph
        else:
            "*** YOUR CODE HERE ***"
            return self.graph.get_output(func)


class DeepQModel(Model):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.

    (We recommend that you implement the RegressionModel before working on this
    part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = .01
        self.hidden = 200

        self.m1 = nn.Variable(4, self.hidden)
        self.b1 = nn.Variable(1, self.hidden)
        self.m2 = nn.Variable(self.hidden, 2)
        self.b2 = nn.Variable(1, 2)

        self.m3 = nn.Variable(2, self.hidden)
        self.b3 = nn.Variable(1, self.hidden)
        self.m4 = nn.Variable(self.hidden, 2)
        self.b4 = nn.Variable(1, 2)

    def run(self, states, Q_target=None):
        """
        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a nn.Graph
        which computes the training loss between your current Q-value
        predictions and these target values, using nn.SquareLoss.

        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        "*** YOUR CODE HERE ***"
        self.graph = nn.Graph([self.m1, self.m2, self.b1, self.b2, self.m3, self.b3, self.m4, self.b4])
        input_s = nn.Input(self.graph, states)
        xm1 = nn.MatrixMultiply(self.graph, input_s, self.m1)
        xm1_plus_b = nn.MatrixVectorAdd(self.graph, xm1, self.b1)

        relu = nn.ReLU(self.graph, xm1_plus_b)

        relum2 = nn.MatrixMultiply(self.graph, relu, self.m2)
        func = nn.MatrixVectorAdd(self.graph, relum2, self.b2)

        if Q_target is not None:
            "*** YOUR CODE HERE ***"
            input_Q = nn.Input(self.graph, Q_target)
            loss = nn.SquareLoss(self.graph, func, input_Q)
            return self.graph
        else:
            "*** YOUR CODE HERE ***"
            return self.graph.get_output(func)

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        self.learning_rate = .001
        self.hidden = 400

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.m1 = nn.Variable(47, self.hidden)
        self.b1 = nn.Variable(1, self.hidden)
        self.m2 = nn.Variable(self.hidden, 5)
        self.b2 = nn.Variable(1, 5)

    def run(self, xs, y=None):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node that represents a (batch_size x hidden_size)
        array, for your choice of hidden_size. It should then calculate a
        (batch_size x 5) numpy array of scores, where higher scores correspond
        to greater probability of the word originating from a particular
        language. You should use `nn.SoftmaxLoss` as your training loss.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)

        Hint: you may use the batch_size variable in your code
        """
        batch_size = xs[0].shape[0]
        h0 = nn.zeros(batch_size)


        "*** YOUR CODE HERE ***"
        self.graph = nn.Graph([self.m1, self.m2, self.b1, self.b2])
        for x in xs:
            input_x = nn.Input(self.graph, x)
            xm1 = nn.MatrixMultiply(self.graph, input_x, self.m1)
            xm1_plus_b = nn.MatrixVectorAdd(self.graph, xm1, self.b1)
            relu = nn.ReLU(self.graph, xm1_plus_b)
            relum2 = nn.MatrixMultiply(self.graph, relu, self.m2)
            func = nn.MatrixVectorAdd(self.graph, relum2, self.b2)

        if y is not None:
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(self.graph, y)
            loss = nn.SquareLoss(self.graph, func, input_y)
            # self.graph.add(loss)
            return self.graph
       
        else:
            "*** YOUR CODE HERE ***"
            return self.graph.get_output(func)
