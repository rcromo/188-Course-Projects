# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

  
        "*** YOUR CODE HERE ***"
  
        current = successorGameState.getScore()
        fVal  = 8.0
        gVal = 9.0

        ghosts = []
        foods = []


        for ghost in newGhostStates:
            ghostPosition = ghost.getPosition()
            ghostDistance = manhattanDistance(newPos, ghostPosition)
            ghosts.append(ghostDistance)


        for food in newFood.asList():
            foodDistance = manhattanDistance(newPos, food)
            foods.append(foodDistance)

        if len(foods):
            fVal = min(foods)


        if ghosts and min(ghosts) != 0:
            gVal = min(ghosts) 


        return current + (8.0/fVal) - (9.0/gVal)




def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        opt = []
     
        def succ(agentIndex, gameState, depth):
            options = []
            if agentIndex == gameState.getNumAgents():
              depth = 1 + depth
              return succ(0,gameState, depth)

            if gameState.isLose() or gameState.isWin() or self.depth == depth:
              return self.evaluationFunction(gameState)

            for act in gameState.getLegalActions(agentIndex):
                options.append(succ(1 + agentIndex, gameState.generateSuccessor(agentIndex, act), depth))

            if agentIndex % gameState.getNumAgents() != 0:
              return min(options)
            else:
              return max(options)

        for action in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0, action)
            temp = succ(1, state, 0)
            opt.append([temp, action])
  
        opt.sort()
        return opt[-1][1]

         


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # opt = []
     
  
        def maxValue(gameState, depth, alpha, beta, agentIndex):
          v = [None,float("-inf")]

          for act in gameState.getLegalActions(agentIndex):
            x = succ(1 + agentIndex, gameState.generateSuccessor(agentIndex, act), depth, alpha, beta)
            x = x[1] 
            maxv = max(v[1], x)


            if maxv > v[1]:
              v = [act, maxv]

            if v[1] > beta:
              return v
            alpha = max(alpha, v[1])
          return v


        def minValue(gameState, depth, alpha, beta, agentIndex):
          v = [None,float("inf")]
          for act in gameState.getLegalActions(agentIndex):
            x = succ(1 + agentIndex, gameState.generateSuccessor(agentIndex, act), depth, alpha, beta)
      
            x = x[1]
            minv = min(v[1], x)

            if minv < v[1]:
              v = [act, minv]

            if v[1] < alpha:
              return v
            beta = min(beta, v[1])
          return v


        def succ(agentIndex, gameState, depth, alpha, beta):
            # options = []
            if agentIndex >= gameState.getNumAgents():
              depth = 1 + depth
              # return succ(0,gameState, depth, alpha, beta)
              agentIndex = 0

            if gameState.isLose() or gameState.isWin() or self.depth == depth:
              return [None, self.evaluationFunction(gameState)]

            # for act in gameState.getLegalActions(agentIndex):
            #     options.append(succ(1 + agentIndex, gameState.generateSuccessor(agentIndex, act), depth, alpha, beta))

            # if agentIndex % gameState.getNumAgents() != 0:
            if agentIndex % gameState.getNumAgents() != 0:
              return minValue(gameState,depth , alpha,beta, agentIndex)
              
            else:
              return maxValue(gameState,depth , alpha,beta, agentIndex)
            

        # for action in gameState.getLegalActions(0):
        #     state = gameState.generateSuccessor(0, action)
        #     temp = succ(1, state, 0, alpha, beta)
        #     opt.append([temp, action])
  
        # #Cannot sort values?
        # opt.sort()
        # return opt[-1][1]
        alpha = float("-inf")
        beta = float("inf")
        temp = succ(0, gameState, 0, alpha, beta)
        return temp[0]
  



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        opt = []
        def succ(agentIndex, gameState, depth):
            options = []
            if agentIndex == gameState.getNumAgents():
              depth = 1 + depth
              return succ(0,gameState, depth)

            if gameState.isLose() or gameState.isWin() or self.depth == depth:
              return self.evaluationFunction(gameState)

            for act in gameState.getLegalActions(agentIndex):
                options.append(succ(1 + agentIndex, gameState.generateSuccessor(agentIndex, act), depth))

            if agentIndex % gameState.getNumAgents() != 0:
              return sum(options)/len(options)
            else:
              return  max(options)

        for action in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0, action)
            temp = succ(1, state, 0)
            opt.append([temp, action])
  
        opt.sort()
        return opt[-1][1]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>

      Very similar to evaluationFunction (gave me 5/6 points). Instead of finding the min
      distance to food, I found the average distance and that did it for the 6/6 points.
      I am adding the currentGameSate.getScore() to the reciprocal of average distance of food 
      multiplied by a weight and also adding the reciprocal of the nearest ghosts multiplied by a weigth.
  
      In the previous evaluationFunction, if the ghost(s) was far from pacman, pacman would stay still
      even if there was food nearby and waited util the ghost approached. This was severely affecting
      the score average. Adding instead of subtracting the recipricated distance of the closest 
      ghost and changing from adding the min distance to food to the average distance, fixed this problem
      for the most part. Now, pacman isn't afraid to approach the ghost and make "risky" moves to get the
      food.
    """
    "*** YOUR CODE HERE ***"
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    # "*** YOUR CODE HERE ***"

    current = currentGameState.getScore()
    fVal  = 8.0
    gVal = 8.0

    ghosts = []
    foods = []

    for ghost in newGhostStates:
        ghostPosition = ghost.getPosition()
        ghostDistance = manhattanDistance(newPos, ghostPosition)
        ghosts.append(ghostDistance)
   
    for food in newFood.asList():
        foodDistance = manhattanDistance(newPos, food)
        foods.append(foodDistance)

    if len(foods):
        fVal = sum(foods)
        fVal = fVal / float(len(foods))

    if ghosts and min(ghosts) != 0:
        gVal = min(ghosts)
        
    return current + (8.0/fVal) + (8.0/gVal) 

# Abbreviation
better = betterEvaluationFunction

