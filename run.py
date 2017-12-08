import gym
from gym import wrappers
import random
from agent import *
from train import *
import argparse

NUM_EPISODES=3000
N_BINS = [8] * 128
MAX_STEPS = 10000
FAIL_PENALTY = -100
EPSILON=0.05
EPSILON_DECAY=1
LEARNING_RATE=0.2
DISCOUNT_FACTOR=0.8

RECORD=True

parser = argparse.ArgumentParser(description="Train and test different reinforcement algorithms on Space Invaders")
# Parse arguments
parser.add_argument("-o", "--observation", type=str, action='store', help="Please specify the observation type (pixel or ram)", required=True)
parser.add_argument("-a", "--agent", type=str, action='store', help="Please specify the agent (qlearn or sarsa or appqlearn)", required=True)

args = parser.parse_args()
print(args)

if args.observation == "pixel":
	env = gym.make('SpaceInvaders-v0')
	#env.configure(remotes=1) # automatically creates a local docker container
	env.reset()
else:
	env = gym.make('SpaceInvaders-ram-v0')
	#env.configure(remotes=1) # automatically creates a local docker container
	env.reset()

if RECORD:
	env = wrappers.Monitor(env, 'spaceinvaders-qlearn-pixel-3000-1', force=True)

def actionFn(env):
    return range(env.action_space.n)

if args.agent == "qlearn":
	agent=QLearningAgent(actionFn(env), 
	                     epsilon=EPSILON, 
	                     alpha=LEARNING_RATE, 
	                     discount=DISCOUNT_FACTOR, 
	                     epsilon_decay=EPSILON_DECAY)
	agent = train(agent, env, args.observation)

elif args.agent == "sarsa":
	agent=SarsaAgent(actionFn(env), 
	                     epsilon=EPSILON, 
	                     alpha=LEARNING_RATE, 
	                     discount=DISCOUNT_FACTOR, 
	                     epsilon_decay=EPSILON_DECAY)
	agent = trainSarsa(agent, env, args.observation)
else:
	agent=ApproximateQLearningAgent(actionFn(env), 
	                     epsilon=EPSILON, 
	                     alpha=LEARNING_RATE, 
	                     discount=DISCOUNT_FACTOR, 
	                     epsilon_decay=EPSILON_DECAY)
	agent = train(agent, env, args.observation)


if RECORD:
	env.close()