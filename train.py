import gym
import numpy
import random
import util
import cv2

NUM_EPISODES=3000
N_BINS = [8] * 100800
MAX_STEPS = 10000
FAIL_PENALTY = -100
EPSILON=0.05
EPSILON_DECAY=1
LEARNING_RATE=0.2
DISCOUNT_FACTOR=0.8

nonflat_min = [[[0]] * 84] * 84
MIN_VALUES = sum(sum(nonflat_min, []), [])
nonflat_max = [[[255]] * 84] * 84
MAX_VALUES = sum(sum(nonflat_max, []), [])
BINS = [numpy.linspace(MIN_VALUES[i], MAX_VALUES[i], N_BINS[i]) for i in xrange(len(MAX_VALUES))]


def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110,:]
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return numpy.reshape(observation,(84,84,1))

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def discretize(obs):
    return tuple([int(numpy.digitize(obs[i], BINS[i])) for i in xrange(len(obs))])


def train(agent, env, obs_type, num_episodes=NUM_EPISODES):
    for i in xrange(NUM_EPISODES):
        print "Episode {}".format(i+1)

        if obs_type == "pixel":
            cur_state = env.reset()
            cur_state = discretize(list(flatten(preprocess(cur_state).tolist())))
        else: 
            cur_state = tuple(env.reset())

        score = 0
        for t in xrange(MAX_STEPS):
            action = agent.getAction(cur_state)
            observation, reward, done, info = env.step(action)
            #print info
            score += reward

            if i % 250 == 0:
                 env.render()
            if obs_type == "pixel":
                next_state = discretize(list(flatten(preprocess(observation).tolist())))
            else:
                next_state = tuple(observation)

            if done:
                agent.update(cur_state, action, next_state, FAIL_PENALTY, done)
                print("Episode finished after {} timesteps. Score: {}".format(t+1, score))
                break

            agent.update(cur_state, action, next_state, reward, done)
            cur_state = next_state

            if t == MAX_STEPS-1:
                print("Episode finished after {} timesteps".format(t+1))

    return agent

def trainSarsa(agent, env, obs_type, num_episodes=NUM_EPISODES):
    for i in xrange(NUM_EPISODES):
        print "Episode {}".format(i+1)

        
        if obs_type == "pixel":
            cur_state = env.reset()
            cur_state = tuple(flatten(preprocess(cur_state).tolist()))
        else: 
            cur_state = tuple(env.reset())
        score = 0
        for t in xrange(MAX_STEPS):
            action = agent.getAction(cur_state)
            observation, reward, done, info = env.step(action)
            #print info
            score += reward

            # renders one game every 50 games. To render more often, decrease number. To not render at all, comment out next two lines
            if i % 50 == 0:
               env.render()

            if obs_type == "pixel":
                next_state = tuple(flatten(preprocess(observation).tolist()))
            else:
                next_state = tuple(observation)
            
            action2 = agent.getAction(next_state)

            if done:
                agent.update(cur_state, action, next_state, action2, FAIL_PENALTY, done)
                print("Episode finished after {} timesteps. Score: {}".format(t+1, score))
                break

            agent.update(cur_state, action, next_state, action2, reward, done)
            cur_state = next_state

            if t == MAX_STEPS-1:
                print("Episode finished after {} timesteps".format(t+1))

    return agent