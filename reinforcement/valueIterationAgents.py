# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()
    

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # print(self.iterations)
        i = 1
        for x in range(self.iterations):
          values = self.values.copy() 
          for s in self.mdp.getStates():
              if not self.mdp.isTerminal(s):
                updatedVals = [self.getQValue(s, action) for action in self.mdp.getPossibleActions(s)]
                values[s] = max(updatedVals)
          self.values = values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        value = 0
        for s, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            value += prob * (self.mdp.getReward(state, action, s) + self.discount * self.values[s])
        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        policy = util.Counter()
        for act in self.mdp.getPossibleActions(state):
            policy[act] = self.getQValue(state, act)
        return policy.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
 
    def runValueIteration(self):
        "*** YOUR CODE HERE ***"


        states = self.mdp.getStates()
        for x in range(self.iterations): 
          state = states[x % len(states)]
          if not self.mdp.isTerminal(state):
            updatedVals = [self.getQValue(state, action) for action in self.mdp.getPossibleActions(state) ]
            self.values[state] = max(updatedVals)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        states = self.mdp.getStates()

        #Compute predecessors of all states
        predDict = dict()
        for state in states:
            predDict[state] = set()

        #initialize empty priority queue
        pQueue = util.PriorityQueue()

        for s in states:
            qVals = util.Counter()
            possibleActions = self.mdp.getPossibleActions(s)
            for a in possibleActions:
                for (next, p) in self.mdp.getTransitionStatesAndProbs(s, a):
                    if p !=0:
                      predDict[next].add(s)
                qVals[a] = self.computeQValueFromValues(s, a)

            #for each non-terminal state s do
            if not self.mdp.isTerminal(s):
              Qmax = qVals[qVals.argMax()]
              diff = abs(self.values[s] - Qmax)
              pQueue.update(s, -diff)

      
        #for iteration in 0, 1, 2, ..., self.iterations - 1, do
        # for i in xrange(self.iterations):
        i = 0
        while i < self.iterations:
            if pQueue.isEmpty():
                return
            s = pQueue.pop()
            if not self.mdp.isTerminal(s):
                qVals = util.Counter()
                possibleActions = self.mdp.getPossibleActions(s)
                for a in possibleActions:
                  qVals[a] = self.computeQValueFromValues(s, a)
                # vals = [self.getQValue(s, action) for action in possibleActions]
                self.values[s] = qVals[qVals.argMax()]
                #for each predecessor p of s, do
                for p in predDict.get(s):
                    possibleActions = self.mdp.getPossibleActions(p)
                    # vals = [self.getQValue(p, action) for action in possibleActions]
                    # print(self.values[p])
                    qValsp = util.Counter()
                    for a in possibleActions:
                      qValsp[a] = self.computeQValueFromValues(p, a)
                    qpMax = qValsp[qValsp.argMax()]
                    diff = abs(self.values[p] - qpMax)

                    if diff > self.theta:
                        pQueue.update(p, -diff)

            i += 1





