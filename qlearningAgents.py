# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        # Dictionary storing all the q-Values for each (state, action) tuple
        self.qvalues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        state_action = (state, action)
        return self.qvalues[state_action]

    def setQValue(self, state, action, value):
        self.qvalues[(state, action)] = value

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        # Retrieve the best action that can be taken in the state
        best_action = self.computeActionFromQValues(state)
        if best_action is not None:
            # Create State Action Tuple
            state_action = (state, best_action)
            # Ask the dictionary
            return self.qvalues[state_action]
        else:
            return 0.0


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # New 'local' qValue counter: action -> qvalue
        qvalues = util.Counter()

        # If there are no legal action we return None
        possible_actions = self.getLegalActions(state)
        if possible_actions is None:
            return None

        # We save our qValues for our possible actions in the local qValue counter
        for action in possible_actions:
            qvalues[action] = self.getQValue(state, action)

        # We pick the best actions, if multiple action have the same qValue then we pick random
        best_actions = []
        highest_value = qvalues[qvalues.argMax()]

        # We save all the actions with the highestValue in bestActions
        for action in qvalues.keys():
            if qvalues[action] == highest_value:
                best_actions.append(action)

        # We return one action, either the only one or a random
        best_action = None

        if len(best_actions) > 1:
            best_action = random.choice(best_actions)
        if len(best_actions) == 1:
            best_action = best_actions[0]

        return best_action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Choose the action that has highest q-value
        action = self.computeActionFromQValues(state)
        "*** YOUR CODE HERE ***"
        # Pick Action
        legal_actions = self.getLegalActions(state)

        if len(legal_actions) == 0:
            return None
        else:
            if util.flipCoin(self.epsilon):
                action = random.choice(legal_actions)
            return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        dscnt = self.discount
        alph = self.alpha
        qval = self.getQValue(state, action)
        next_value = self.computeValueFromQValues(nextState)

        new_value = (1 - alph) * qval + alph * (reward + dscnt * next_value)

        self.setQValue(state, action, new_value)


class PacmanQAgent(QLearningAgent):
    """Exactly the same as QLearningAgent, but with different default parameters"""

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        all_features = self.featExtractor.getFeatures(state, action)
        qval = 0
        for feature in all_features:
            qval += self.weights[feature] * all_features[feature]
        return qval

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        all_features = self.featExtractor.getFeatures(state, action)
        difference = reward + self.discount * self.getValue(nextState) - self.getQValue(state, action)
        for feature in all_features:
            self.weights[feature] += self.alpha * difference * all_features[feature]


def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print "weights: "
            print self.weights
