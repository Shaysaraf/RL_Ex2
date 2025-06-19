# RL_Ex2
Reinforcment Learning Course - Ex2 Frozen Lake
# FrozenLake MDP: Value Iteration and Policy Iteration

This repository implements **Value Iteration** and **Policy Iteration** algorithms to solve the `FrozenLake-v1` environment from OpenAI Gym. These classic dynamic programming techniques are used to compute optimal policies for Markov Decision Processes (MDPs) by estimating value functions and improving policies iteratively.

---

## 📁 Project Structure

.
├── value_iter.py # Value Iteration implementation and visualization
├── policy_iter.py # Policy Iteration implementation and visualization
├── README.md # Documentation

---

## 🧠 Algorithms Overview

### 🔹 Value Iteration (`value_iter.py`)
- Computes value functions using the Bellman optimality equation.
- Extracts a sequence of greedy policies based on updated values.
- Terminates when value updates converge below a specified threshold.
- Visualizations:
  - Value function grid per iteration.
  - Policy arrows on a 4x4 grid per iteration.
  - State value convergence plot.

### 🔹 Policy Iteration (`policy_iter.py`)
- Alternates between:
  - **Policy Evaluation**: Solves a system of linear equations to evaluate the current policy.
  - **Policy Improvement**: Computes a new greedy policy from the current value function.
- Terminates when the policy is stable (no changes in improvement step).
- Visualizations:
  - Value and policy grids per iteration.
  - State value convergence plot.

---

## 📈 Outputs

Each script generates:
- A sequence of printed **value functions** (`V`) as 4×4 grids.
- Corresponding **policies** displayed as arrow directions:
  - `←` (West), `↓` (South), `→` (East), `↑` (North)
- A plot showing the **convergence of each state value** over iterations.

These outputs are essential for analyzing the learning process and verifying convergence.

---

## 🛠️ Requirements

- Python 3.7+
- `numpy`
- `matplotlib`
- `gym`

Install the required dependencies using:

```bash
pip install numpy matplotlib gym
▶️ How to Run
Value Iteration
bash
Copy
Edit
python value_iter.py
Policy Iteration
bash
Copy
Edit
python policy_iter.py
Each script will:

Print iteration logs to the console.

Display the evolving policy and value function.

Generate convergence plots using Matplotlib.

📝 Submission Requirements
This implementation satisfies the following deliverables:

✅ value_iter.py: Contains complete value iteration algorithm and visualizations.

✅ policy_iter.py: Contains complete policy iteration algorithm and visualizations.

✅ Iteration tables and value/policy outputs printed to the console.

✅ Matplotlib plots:

(i) Optimal policy at each iteration.

(ii) Value convergence per state over iterations.

📌 Notes
Ties in argmax are broken by selecting the action with the lowest index, ensuring deterministic behavior.

Value iteration uses non-in-place updates, adhering to synchronous update rules (as per the assignment guidelines).

Random seed is fixed for reproducibility of results.
