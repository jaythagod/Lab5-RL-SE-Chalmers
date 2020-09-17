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
        self.valueIteration()

        # Write value iteration code here
        "*** YOUR CODE HERE ***"


    def valueIteration(self):
        states = self.mdp.getStates()

        for iteration in range(0, self.iterations):
            # initialise all states to 0
            computed_q_values = util.Counter()
            for state in states:
                # get best action in the state according the policy
                action = self.getAction(state)
                if action is not None:
                    # compute Q(s,a)
                    computed_q_values[state] = self.getQValue(state, action)
            self.values = computed_q_values

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
        value_we_need_to_return = 0
        nextstate_prob_tup_list = self.mdp.getTransitionStatesAndProbs(state, action)

        for nxt, prob in nextstate_prob_tup_list:
            reward = self.mdp.getReward(state, action, nxt)
            value = self.getValue(nxt)
            # calculate Qopt(s,a)
            value_we_need_to_return += prob * (reward + self.discount * value)

        return value_we_need_to_return
        
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        max_val = float("-inf")
        action_with_max_q_value = None
        possible_actions = self.mdp.getPossibleActions(state)
        if len(possible_actions) == 0:
            return action_with_max_q_value
        else:
            for action in possible_actions:
                q_value = self.computeQValueFromValues(state, action)
                if q_value > max_val:
                    max_val = q_value
                    action_with_max_q_value = action
            return action_with_max_q_value

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
