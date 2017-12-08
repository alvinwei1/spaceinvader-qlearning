# spaceinvader-qlearning
## Prerequisites

To install all the prerequisites for code use, open the terminal and input:

```
git clone https://github.com/alvinwei1/spaceinvader-qlearning
cd spaceinvader-qlearning
pip install -r requirements.txt
```
## Running the tests
The code below takes Space Invaders as either a pixel image or RAM input for its observations and trains one of three reinforcement learning agents to play the game. A directory is created containing a \verb|json| file of the episode lengths and rewards and \verb|mp4| files recording the simulation of the agent playing. To run the code use:
```
usage: run.py [-h] -o OBSERVATION -a AGENT
```

The ```AGENT``` argument is the type of reinforcement agent ```qlearn```, ```sarsa```, or ```appqlearn```), representing the Q-Learning, SARSA, and Approximate Q-Learning agents classes to use, respectively.

For instance, to run a Q-Learning agent taking Space Invaders as RAM input run:
```
python run.py -o ram -a qlearn
```
You can see the output in a directory specified in line 41 of ```run.py```. 

For example:

```env = wrappers.Monitor(env, 'spaceinvaders-experiment-ram-sarsa', force=True)```

saves the output in the directory ```spaceinvaders-noreward-experiment-ram-sarsa``` and overwrites any previous output in that directory. The statistics file is named ```openaigym.episode_batch.0.28847.stats.json```.
