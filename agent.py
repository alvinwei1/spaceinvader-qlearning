import gym
import numpy
import random
import util

N_BINS = [8] * 128
MAX_STEPS = 10000
FAIL_PENALTY = -100
EPSILON=0.05
EPSILON_DECAY=1
ALPHA=0.2
DISCOUNT=0.8


class QLearningAgent:
    """
      Q-Learning Agent
    """
    def __init__(self, actionFn, epsilon=0.5, alpha=0.5, discount=0.9, epsilon_decay=1, **args):
        self.epsilon = epsilon
        self.alpha = alpha
        self.discount = discount
        self.epsilon_decay=epsilon_decay
        self.legal_actions = actionFn
        self.q_values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
        """
        return self.q_values[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,legal_actions)
        """
        values = []
        actions = self.legal_actions
        if not actions:
            values.append(0.0)
        for action in actions:
                values.append(self.getQValue(state, action))
            
        return max(values)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.
        """
        legal_q_values = util.Counter()
        actions = self.legal_actions
        if not actions:
            return None
        else:
            for action in actions:
                legal_q_values[(state, action)] = self.q_values[(state, action)]

        # If there is more than one max q-value, randomly choose one of them
        best_keys = []
        for key in legal_q_values.sortedKeys():
            if legal_q_values[key] == legal_q_values[legal_q_values.argMax()]:
                best_keys.append(key)
            else:
                break

        return random.choice(best_keys)[1]

    def getAction(self, state):
        """
          Compute the action to take in the current state.
        """
        # Pick Action
        legalActions = self.legal_actions
        action = None
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward, done):
        """
          Train calls this to observe a
          state = action => nextState and reward transition.
        """
        if done:
            reward = FAIL_PENALTY
            self.q_values[(state, action)] = (1.0 - self.alpha) * self.getQValue(state, action) + self.alpha * (reward)
        else:
            self.q_values[(state, action)] = (1.0 - self.alpha) * self.getQValue(state, action) + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)



class SarsaAgent:
    """
      Sarsa Agent
    """
    def __init__(self, actionFn, epsilon=0.5, alpha=0.5, discount=0.9, epsilon_decay=1, **args):
        self.epsilon = epsilon
        self.alpha = alpha
        self.discount = discount
        self.epsilon_decay=epsilon_decay
        self.legal_actions = actionFn
        self.q_values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
        """
        return self.q_values[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,legal_actions)
        """
        values = []
        actions = self.legal_actions
        if not actions:
            values.append(0.0)
        for action in actions:
                values.append(self.getQValue(state, action))
            
        return max(values)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.
        """
        legal_q_values = util.Counter()
        actions = self.legal_actions
        if not actions:
            return None
        else:
            for action in actions:
                legal_q_values[(state, action)] = self.q_values[(state, action)]

        # If there is more than one max q-value, randomly choose one of them
        best_keys = []
        for key in legal_q_values.sortedKeys():
            if legal_q_values[key] == legal_q_values[legal_q_values.argMax()]:
                best_keys.append(key)
            else:
                break

        return random.choice(best_keys)[1]

    def getAction(self, state):
        """
          Compute the action to take in the current state.
        """
        # Pick Action
        legalActions = self.legal_actions
        action = None
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action1, nextState, action2, reward, done):
        """
          Train calls this to observe a
          state = action => nextState and reward transition.
        """
        q_next = self.getQValue(nextState, action2)
        if done:
            reward = FAIL_PENALTY
            self.q_values[(state, action1)] = reward
        else:
            self.q_values[(state, action1)] = (1.0 - self.alpha) * self.getQValue(state, action1) + self.alpha * (reward + self.discount * q_next)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class ApproximateQLearningAgent(QLearningAgent):
    """
       ApproximateQLearningAgent
    """
    def __init__(self, actionFn, epsilon=0.5, alpha=0.5, discount=0.9, epsilon_decay=1, **args):
        self.epsilon = epsilon
        self.alpha = alpha
        self.discount = discount
        self.epsilon_decay=epsilon_decay
        self.legal_actions = actionFn
        self.weights = util.Counter()
        self.q_values = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Return Q(state,action) = w * featureVector
        """
        features = [elem for elem in state]
        features.append(1)

        sum = 0
        for i in xrange(len(features)):
            sum += self.weights[i] * features[i]
        return sum

    def update(self, state, action, nextState, reward, done):
        """
           Update weights based on transition
           weights = counter where keys are index of np array 
           features = numpy array 
        """
        newValue = reward + self.discount * self.computeValueFromQValues(nextState)
        oldValue = self.getQValue(state, action)
        difference = newValue - oldValue

        features = [elem for elem in state]
        features.append(1)
        for i in xrange(len(features)):
          self.weights[i] += self.alpha * difference * features[i]
        self.weights.normalize()