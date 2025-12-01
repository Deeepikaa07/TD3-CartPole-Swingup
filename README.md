# 🌟 TD3 on CartPole-Swingup — Reinforcement Learning 

This repository contains an implementation of the **Twin Delayed Deep Deterministic Policy Gradient (TD3)** algorithm applied to the **CartPole-Swingup** task from the **DeepMind Control Suite (DMC)**.  
The assignment requires training with seeds **0, 1, 2**, evaluating with seed **10**, and plotting the **mean ± standard deviation** learning curve.

## 🚀 Project Overview

The CartPole-Swingup environment starts with the pole **hanging downward**, requiring the agent to:
- Pump energy to swing the pole up  
- Balance it near the upright position  

TD3 is ideal for this task because of its:
- 🟦 **Twin Q-networks** (fixes overestimation)
- 🔁 **Delayed policy updates** (more stable learning)
- 🎯 **Target policy smoothing** (robust critic targets)
- 🧭 **Deterministic actor + action noise** (controlled exploration)


## 🧠 Algorithm: TD3

TD3 (Fujimoto et al., 2018) is a deterministic actor–critic method for continuous actions.

### 🔑 Key Features
- Twin Q-functions  
- Policy smoothing with Gaussian noise  
- Clipped noise for stability  
- Actor updates less frequently than critic  
- NormalActionNoise for exploration  


## 🛠️ Installation

conda create -n RLassign python=3.10
conda activate RLassign

pip install stable-baselines3[extra]
pip install gymnasium "gymnasium[other]"
pip install dm_control seaborn matplotlib moviepy


🎯 Training

Run:

python train.py


This will:

Train TD3 with seeds 0, 1, 2

Save logs → logs/seed_X.monitor.csv

Save models → weights/td3_cartpole_seedX.zip

📝 Training Snippet
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.1 * np.ones(n_actions)
)

model = TD3(
    "MultiInputPolicy",
    env,
    action_noise=action_noise,
    seed=seed,
    device="cuda:0",
    verbose=1,
)
model.learn(total_timesteps=200000)

## 🎥 TD3 Evaluation Videos (Seed 10)

### ▶️ TD3 Tested with CartPole Seed 10 (Trained Seed 0)
https://github.com/Deeepikaa07/TD3-CartPole-Swingup/issues/1#issue-3678741887

### ▶️ TD3 Tested with CartPole Seed 10 (Trained Seed 1)
https://github.com/Deeepikaa07/TD3-CartPole-Swingup/issues/2#issue-3678743926

### ▶️ TD3 Tested with CartPole Seed 10 (Trained Seed 2)
https://github.com/Deeepikaa07/TD3-CartPole-Swingup/issues/3#issue-3678746581 


📊 Learning Curve

You can generate the mean ± std plot using:

plot.ipynb

### Final TD3 Learning Curve

<div align="center">
  <img src="td3_eval.png" width="70%">
</div>

---

## ⚙️ Hyperparameters

| Parameter | Value |
|----------|--------|
| Algorithm | TD3 |
| Learning Rate | 3e-4 |
| Batch Size | 256 |
| Buffer Size | 1e6 |
| Discount (γ) | 0.99 |
| Tau | 0.005 |
| Policy Noise | 0.2 |
| Noise Clip | 0.5 |
| Action Noise | 0.1 |
| Network Architecture | [256, 256] |
| Activation | ReLU |
| Timesteps | 200,000 per seed |

## 📈 Results Summary

✅ TD3 successfully swings up and stabilizes the pole

📉 Standard deviation shrinks over time → stable learning

🎯 Final performance stabilizes around 600–650 reward

⚠️ Seed-to-seed variability is expected in deterministic actor–critic algorithms

🤖 Despite SAC performing better typically, TD3 performed strongly on seeds 0 & 2

## 📚 References

Fujimoto, S., van Hoof, H., & Meger, D.
Addressing Function Approximation Error in Actor-Critic Methods. ICML, 2018.

Stable-Baselines3 TD3 Documentation
https://stable-baselines3.readthedocs.io/en/master/modules/td3.html