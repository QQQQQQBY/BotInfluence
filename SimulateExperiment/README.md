<div align="center">
  <img src="/Users/boyuqiao/Downloads/BotInfluence/SimulateExperiment/Figure/GPT4oGeneratedfigure.png" alt="Simulate Disinformation Dissemination" height="375">
</div>

<p align="center">
  🧟Simulate Disinformation Dissemination🧟
</p>


## 🐣 模拟虚假信息的传播

We have designed three types of agents: malicious bot agents, legitimate bot agents and ordinary user agents. These three types of agents participate in the information propagation process through the information propagation simulation network we have constructed.

🐡 First, we assign explicit activation timesteps to ordinary users, malicious bots, and legitimate bots based on a Bernoulli distribution and to ensure that at least one bot is activated at the initial timestep by executing the scripts `python DataPreprocess/preprocess_activation_time_step.py`.

🪸 Second, we run `python conclude_humantext.py` operations to summarize the user's historical text information to assist the subsequent ordinary user agent in generating text content that conforms to the behavioral pattern.

🦋 Finally, we initiate the simulation by executing `python main.py`.  The detailed algorithmic workflow is as follows:


（1）分别构建存储普通用户代理、恶意机器人代理和合法机器人代理历史行为及其属性变化的文件，分别为`Dataset/RecordedDataset/HumanDataset/`, `Dataset/RecordedDataset/MBotDataset`, `Dataset/RecordedDataset/LBotDataset/`.

（2）依据时间步的变化，用户在激活时间步内浏览信息并做出是否相信浏览到的信息、是否传播等决策。针对第一个时间步，我们需要特殊处理，即第一个时间步，普通用户即使激活也不会有行动，因为其还未接触到虚假信息相关的内容，第一个时间步只由被激活的恶意机器人节点展开信息的扩散。

（3）