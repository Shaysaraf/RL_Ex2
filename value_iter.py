##################################
# Create env
import gym
env = gym.make('FrozenLake-v1', render_mode='human')
env = env.unwrapped  # <-- allows access to internal MDP info
print(env.__doc__)
print("")

#################################
# Some basic imports and setup
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
class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        self.P = P # state transition and reward probabilities
        self.nS = nS # number of states
        self.nA = nA # number of actions
        self.desc = desc # grid description (for plotting)

# ✅ FIXED HERE:
P = env.P
nS = env.observation_space.n
nA = env.action_space.n
desc = env.desc if hasattr(env, "desc") else None

mdp = MDP({s: {a: [tup[:3] for tup in tups] for a, tups in a2d.items()} for s, a2d in P.items()}, nS, nA, desc)

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
# Programming Question No. 1 - Value Iteration

def value_iteration(mdp, gamma, nIt):
    print("Iteration | max|V-Vprev| | # chg actions | V[0]")
    print("----------+--------------+---------------+---------")
    Vs = [np.zeros(mdp.nS)]
    pis = []
    for it in range(nIt):
        oldpi = pis[-1] if pis else None
        Vprev = Vs[-1]
        V = np.zeros(mdp.nS)
        pi = np.zeros(mdp.nS, dtype=int)
        for s in range(mdp.nS):
            q_sa = np.zeros(mdp.nA)
            for a in range(mdp.nA):
                for (prob, next_state, reward) in mdp.P[s][a]:
                    q_sa[a] += prob * (reward + gamma * Vprev[next_state])
            V[s] = np.max(q_sa)
            pi[s] = np.argmax(q_sa)
        max_diff = np.abs(V - Vprev).max()
        nChgActions = "N/A" if oldpi is None else (pi != oldpi).sum()
        print("%4i      | %6.5f      | %4s          | %5.3f" % (it, max_diff, nChgActions, V[0]))
        Vs.append(V)
        pis.append(pi)
    return Vs, pis

GAMMA = 0.95
Vs_VI, pis_VI = value_iteration(mdp, gamma=GAMMA, nIt=50)

#################################
# Visualization of Value Iteration

for (V, pi) in zip(Vs_VI[:10], pis_VI[:10]):
    plt.figure(figsize=(3,3))
    plt.imshow(V.reshape(4,4), cmap='gray', interpolation='none', clim=(0,1))
    ax = plt.gca()
    ax.set_xticks(np.arange(4)-.5)
    ax.set_yticks(np.arange(4)-.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    Y, X = np.mgrid[0:4, 0:4]
    a2uv = {0: (-1, 0), 1:(0, -1), 2:(1,0), 3:(0, 1)}
    Pi = pi.reshape(4,4)
    for y in range(4):
        for x in range(4):
            a = Pi[y, x]
            u, v = a2uv[a]
            plt.arrow(x, y, u*.3, -v*.3, color='m', head_width=0.1, head_length=0.1)
            plt.text(x, y, str(env.desc[y,x].item().decode()),
                     color='g', size=12, verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
    plt.grid(color='b', lw=2, ls='-')

# Plot each state's value as a function of iteration
Vs_array = np.array(Vs_VI)  # Shape: (nIt+1, nS)
plt.figure(figsize=(10,6))
for s in range(mdp.nS):
    plt.plot(range(len(Vs_array)), Vs_array[:, s], label=f"State {s}")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.title("State Values Over Iterations (Value Iteration)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()
