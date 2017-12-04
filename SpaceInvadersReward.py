import gym
from gym import wrappers
import cv2
import numpy
import random

NUM_EPISODES=3000
N_BINS = [8] * 100800
MAX_STEPS = 10000
FAIL_PENALTY = -100
EPSILON=0.5
EPSILON_DECAY=0.99
LEARNING_RATE=0.05
DISCOUNT_FACTOR=0.9

RECORD=True

nonflat_min = [[[0]] * 84] * 84
MIN_VALUES = sum(sum(nonflat_min, []), [])
nonflat_max = [[[255]] * 84] * 84
MAX_VALUES = sum(sum(nonflat_max, []), [])
BINS = [numpy.linspace(MIN_VALUES[i], MAX_VALUES[i], N_BINS[i]) for i in xrange(len(MAX_VALUES))]


class QLearningAgent:

  def __init__(self, legal_actions_fn, epsilon=0.5, alpha=0.5, gamma=0.9, epsilon_decay=1):
    """
    args
      legal_actions_fn    takes a state and returns a list of legal actions
      alpha       learning rate
      epsilon     exploration rate
      gamma       discount factor
    """
    self.epsilon = epsilon
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon_decay=epsilon_decay
    self.legal_actions_fn = legal_actions_fn

    # map: {(state, action): q-value}
    self.q_values = {}
    # map: {state: action}
    self.policy = {}
    

  def get_value(self, s):
    a = self.get_action(s)
    return self.get_qvalue(s, a)


  def get_qvalue(self, s, a):
    if (s,a) in self.q_values:
      return self.q_values[(s,a)]
    else:
      # set to 0
      self.q_values[(s,a)] = 0
      return 0

  def _set_qvalue(self, s, a, v):
    self.q_values[(s,a)] = v


  def get_optimal_action(self, state):
    legal_actions = self.legal_actions_fn(state)
    assert len(legal_actions) > 0, "no legal actions"
    if state in self.policy:
      return self.policy[state]
    else:
      # randomly select an action as default and return
      self.policy[state] = legal_actions[numpy.random.randint(0, len(legal_actions))]
      return self.policy[state]

  
  def get_action(self, state):
    """
    Epsilon-greedy action
    args
      state           current state      
    returns
      an action to take given the state
    """
    legal_actions = self.legal_actions_fn(state)

    assert len(legal_actions) > 0, "no legal actions on state {}".format(state)

    if numpy.random.random() < self.epsilon:
      # act randomly
      self.epsilon = self.epsilon*self.epsilon_decay
      return legal_actions[numpy.random.randint(0, len(legal_actions))]
    else:
      if state in self.policy:
        return self.policy[state]
      else:
        # set the first action in the list to default and return
        self.policy[state] = legal_actions[0]
        return legal_actions[0]


  def learn(self, s, a, s1, r, is_done):
    """
    Updates self.q_values[(s,a)] and self.policy[s]
    args
      s         current state
      a         action taken
      s1        next state
      r         reward
      is_done   True if the episode concludes
    """
    # update q value
    if is_done:
      sample = r
    else:
      sample = r + self.gamma*max([self.get_qvalue(s1,a1) for a1 in self.legal_actions_fn(s1)])
    
    q_s_a = self.get_qvalue(s,a)
    q_s_a = q_s_a + self.alpha*(sample - q_s_a)
    self._set_qvalue(s,a,q_s_a)

    # policy improvement
    legal_actions = self.legal_actions_fn(s)
    s_q_values = [self.get_qvalue(s,a) for a in legal_actions]
    self.policy[s] = legal_actions[s_q_values.index(max(s_q_values))]


def preprocess(observation):
  observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
  observation = observation[26:110,:]
  ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
  return numpy.reshape(observation,(84,84,1))

def discretize(obs):
  # print('obs len: ', len(obs))
  # print('BINS len: ', len(BINS))
  return tuple([int(numpy.digitize(obs[i], BINS[i])) for i in xrange(len(obs))])

def flatten(container):
  for i in container:
    if isinstance(i, (list,tuple)):
      for j in flatten(i):
          yield j
    else:
      yield i

def train(agent, env, history, num_episodes=NUM_EPISODES):
  for i in xrange(NUM_EPISODES):
    if i % 100:
      print "Episode {}".format(i+1)

    obs = env.reset()
    # print(obs) 
    # print(len(obs))
    # if type(obs) is list:
    preprocess_obs = preprocess(obs)

    obs_flat = list(flatten(preprocess_obs.tolist()))
    #print('obs len:', len(obs))
    #print('obs_flat len:', len(obs_flat))
    cur_state = discretize(obs_flat)
    # else:
    # cur_state = discretize(obs)
    lives = 3

    for t in xrange(MAX_STEPS):
      action = agent.get_action(cur_state)
      #print('action: ', action)
      observation, reward, done, info = env.step(action)

      if (action == 1) or (action == 4) or (action == 5):
        reward += 10

      env.render()

      # if type(observation) is list:
      preprocess_observation = preprocess(observation)
      observation_flat = list(flatten(preprocess_observation.tolist()))
      next_state = discretize(observation_flat)
      # else:
        # next_state = discretize(observation)

      if done:
        reward = FAIL_PENALTY
        agent.learn(cur_state, action, next_state, reward, done)
        print("Episode finished after {} timesteps".format(t+1))
        history.append(t+1)
        break

      agent.learn(cur_state, action, next_state, reward, done)
      cur_state = next_state

      if t == MAX_STEPS-1:
        history.append(t+1)
        print("Episode finished after {} timesteps".format(t+1))

  return agent, history

env = gym.make('SpaceInvaders-v0')
#env.configure(remotes=1) # automatically creates a local docker container
observation_n = env.reset()

if RECORD:
  env = gym.wrappers.Monitor(env, '/tmp/spaceinvaders-experiment-3', force=True)

def get_actions(state):
  #return [0, 1, 3, 4, 11, 12]
  return range(6)

agent=QLearningAgent(get_actions, 
                     epsilon=EPSILON, 
                     alpha=LEARNING_RATE, 
                     gamma=DISCOUNT_FACTOR, 
                     epsilon_decay=EPSILON_DECAY)

history = []

agent, history = train(agent, env, history)

if RECORD:
  env.monitor.close()