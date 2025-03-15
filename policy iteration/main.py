import gym
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns

env = gym.make('FrozenLake-v1')
env.reset()
env.render()
env.close()

import numpy as np

def policyEvaluation(valueFunctionVector, max_iter, tolerance, discountFactor):
    convergenceTrack = []  # Initialize convergence tracking
    env_states = env.observation_space.n  # Number of states

    for i in range(max_iter):
        convergenceTrack.append(np.linalg.norm(valueFunctionVector, 2))
        valueFunctionVectorNew = np.zeros(env_states)  
        
        for state in range(env_states): 
            outer_sum = 0
            if state in env.P:  
                for action in env.P[state]:
                    inner_sum = 0
                    for prob, next_state, reward, done in env.P[state][action]:
                        inner_sum += prob * (reward + discountFactor * valueFunctionVector[next_state])
                    outer_sum += (1 / len(env.P[state])) * inner_sum  

            valueFunctionVectorNew[state] = outer_sum 

        if np.max(np.abs(valueFunctionVectorNew - valueFunctionVector)) < tolerance:
            print('Converged!')
            return valueFunctionVectorNew

        valueFunctionVector = valueFunctionVectorNew.copy()  

    return valueFunctionVectorNew  


discountFactor = 0.9
maxNumberOfIterations = 1000
convergenceTolerance = 1e-6
valueFunctionVector = np.zeros(env.observation_space.n)

print("Value Function:", policyEvaluation(valueFunctionVector, maxNumberOfIterations, convergenceTolerance, discountFactor))


policy = np.zeros(env.observation_space.n, dtype=int)
for state in range(env.observation_space.n):
    neighbors = [
        state - 1 if state % 4 > 0 else -1,        # left
        state + 4 if state < 12 else -1,           # down
        state + 1 if state % 4 < 3 else -1,        # right
        state - 4 if state >= 4 else -1            # up
    ]

    neighbor_values = [
        valueFunctionVector[n] if n >= 0 else -float('inf')
        for n in neighbors
    ]
    
    policy[state] = np.argmax(neighbor_values)

print(policy)


def play_game(env, policy, max_steps=100):
    state = env.reset()
    total_reward = 0
    for step in range(max_steps):
        env.render()
        action = policy[state]
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
        time.sleep(0.5)
        if done:
            print(f"Episode finished after {step + 1} steps. Total reward: {total_reward}")
            break
    env.close()

print("Starting the game with the neighbor-based policy...")
play_game(env, policy)
