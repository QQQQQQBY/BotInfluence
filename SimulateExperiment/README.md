<div align="center">
  <img src="/Users/boyuqiao/Desktop/BotInfluence-main/BotInfluence/SimulateExperiment/Figure/GPT4oGeneratedfigure.png" alt="Simulate Disinformation Dissemination" height="375">
</div>

<p align="center">
  🧟Simulate Disinformation Dissemination🧟
</p>


## 🐣 Simulate the spread of disinformation

We have designed three types of agents: malicious bot agents, legitimate bot agents and ordinary user agents. These three types of agents participate in the information propagation process through the information propagation simulation network we have constructed.

🐡 First, we assign explicit activation timesteps to ordinary users, malicious bots, and legitimate bots based on a Bernoulli distribution and to ensure that at least one bot is activated at the initial timestep by executing the scripts `python DataPreprocess/preprocess_activation_time_step.py`.

🪸 Second, we run `python conclude_humantext.py` operations to summarize the user's historical text information to assist the subsequent ordinary user agent in generating text content that conforms to the behavioral pattern.

🦋 Finally, we initiate the simulation by executing `python main.py`.  The detailed algorithmic workflow is as follows:

<div align="center">
  <img src="/Users/boyuqiao/Desktop/BotInfluence-main/BotInfluence/SimulateExperiment/Figure/Algorithm.jpg" alt="Simulate Disinformation Dissemination Algorithm" height="875">
</div>