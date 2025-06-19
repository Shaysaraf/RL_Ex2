##################################
# Create env
import gym
env = gym.make('FrozenLake-v1', render_mode='human')
env = env.unwrapped
print(env.__doc__)
print("")

#################################
# Some basic imports and setup
# Let's look at what a random episode looks like.

import numpy as np, numpy.random as nr, gym
import matplotlib.pyplot as plt
#%matplotlib inline
np.set_printoptions(precision=3)

# Seed RNGs so you get the same printouts as me
env.reset(seed=0); np.random.seed(10)
# Generate the episode
env.reset()
for t in range(100):
    env.render()
    a = env.action_space.sample()
    ob, rew, terminated, truncated, info = env.step(a)
    done = terminated or truncated
    if done:
        break
assert done
env.render();

#################################
# Create MDP for our env
# We extract the relevant information from the gym Env into the MDP class below.
# The `env` object won't be used any further, we'll just use the `mdp` object.

class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        self.P = P # state transition and reward probabilities
        self.nS = nS # number of states
        self.nA = nA # number of actions
        self.desc = desc # grid description (for plotting)

P = env.P
nS = env.observation_space.n
nA = env.action_space.n
desc = env.desc if hasattr(env, "desc") else None

mdp = MDP({s: {a: [tup[:3] for tup in tups] for a, tups in a2d.items()} for s, a2d in P.items()}, nS, nA, desc)
GAMMA = 0.95 # we'll be using this same value in subsequent problems

print("")
print("mdp.P is a two-level dict where the first key is the state and the second key is the action.")
print("The 2D grid cells are associated with indices [0, 1, 2, ..., 15] from left to right and top to down, as in")
print(np.arange(16).reshape(4,4))
print("Action indices [0, 1, 2, 3] correspond to West, South, East and North.")
print("mdp.P[state][action] is a list of tuples (probability, nextstate, reward).\n")
print("For example, state 0 is the initial state, and the transition information for s=0, a=0 is \nP[0][0] =", mdp.P[0][0], "\n")
print("As another example, state 5 corresponds to a hole in the ice, in which all actions lead to the same state with probability 1 and reward 0.")
for i in range(4):
    print("P[5][%i] =" % i, mdp.P[5][i])
print("")

#################################
# Programing Question No. 2, part 1 - implement where required.

#################################
# Part 1: Compute V^pi

def compute_vpi(pi, mdp, gamma):
    A = np.eye(mdp.nS)
    b = np.zeros(mdp.nS)
    for s in range(mdp.nS):
        a = pi[s]
        for prob, next_s, reward in mdp.P[s][a]:
            A[s, next_s] -= gamma * prob
            b[s] += prob * reward
    V = np.linalg.solve(A, b)
    return V

# Example test
actual_val = compute_vpi(np.arange(16) % mdp.nA, mdp, gamma=GAMMA)
print("Policy Value: ", actual_val)


actual_val = compute_vpi(np.arange(16) % mdp.nA, mdp, gamma=GAMMA)
print("Policy Value: ", actual_val)

#################################
# Programing Question No. 2, part 2 - implement where required.
#################################
# Part 2: Compute Q^pi

def compute_qpi(vpi, mdp, gamma):
    Qpi = np.zeros([mdp.nS, mdp.nA])
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            for prob, next_s, reward in mdp.P[s][a]:
                Qpi[s, a] += prob * (reward + gamma * vpi[next_s])
    return Qpi

# Example test
Qpi = compute_qpi(np.arange(mdp.nS), mdp, gamma=GAMMA)
print("Policy Action Value: ", Qpi)


Qpi = compute_qpi(np.arange(mdp.nS), mdp, gamma=0.95)
print("Policy Action Value: ", actual_val)

#################################
# Programing Question No. 2, part 3 - implement where required.
# Policy iteration

#################################
# Part 3: Policy Iteration

def policy_iteration(mdp, gamma, nIt):
    Vs = []
    pis = []
    pi_prev = np.zeros(mdp.nS, dtype='int')
    pis.append(pi_prev)
    print("Iteration | # chg actions | V[0]")
    print("----------+---------------+---------")
    for it in range(nIt):
        vpi = compute_vpi(pi_prev, mdp, gamma)
        qpi = compute_qpi(vpi, mdp, gamma)
        pi = qpi.argmax(axis=1)
        print("%4i      | %6i        | %6.5f"%(it, (pi != pi_prev).sum(), vpi[0]))
        Vs.append(vpi)
        pis.append(pi)
        if (pi == pi_prev).all():  # converged
            break
        pi_prev = pi
    return Vs, pis

# Run policy iteration
Vs_PI, pis_PI = policy_iteration(mdp, gamma=GAMMA, nIt=20)

action_arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
for epoch, (V, pi) in enumerate(zip(Vs_PI, pis_PI[1:])):  # skip the initial pi
    print(f"Iteration {epoch}:")
    print("Value function:")
    print(np.array2string(np.array(V).reshape(4, 4), precision=3, suppress_small=True))
    print("Policy:")
    pi_grid = np.array(pi).reshape(4, 4)
    arrow_grid = np.vectorize(action_arrows.get)(pi_grid)
    for row in arrow_grid:
        print(' '.join(row))
    print()


#################################
# Plotting value convergence

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
for s in range(mdp.nS):
    plt.plot([v[s] for v in Vs_PI], label=f"state {s}")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.title("State Values Over Policy Iteration (Policy Iteration)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.show()
