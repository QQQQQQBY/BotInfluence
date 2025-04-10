<div align="center">
  <img src="/Users/boyuqiao/Downloads/BotInfluence/DisseminationNetwork/Figure/GPT4oGeneratedFigure.png" alt="Construct Disinformation Netwerk" height="375">
</div>

<p align="center">
  🕸️Construct Disinformation Netwerk🕸️
</p>

## 🍅 Construct Disinformation Dissemination Simulation Network

We used the **random block model** (SBM) and the **Barabasi-Albert** (BA) model (collectively SB-BA) to construct the potential disinformation dissemination network and simulate the information propagation structure in the real social network. 

**SBM** introduces the community structure, capturing the characteristics of dense connections within the community and sparse connections between the communities, while **BA** simulates the priority connection mechanism and scale-free characteristics common in social networks.

🌶️ **Construction algorithm process**:


---

**Algorithm: Disinformation Dissemination Network Construction (BA-SBM)**
**Inputs:**

- Total number of nodes: $N$  
- Initial number of nodes in each community: $m_0$  
- Number of communities: $K$  
- User interest community scores: $x_{ij}$  
- Predefined threshold for community assignment: $\tau$  
- Social influence scores for nodes: $S_i$  
- Number of edges each new node forms: $m \ (m \le m_0)$  
- Community label(s) of each user $i$: $\{c_j^i\}$

**Outputs:**

- Graph $G(V, E)$ representing the constructed BA-SBM network

**(1) Node Community Assignment (SBM Component)**

For each initial node $i = 1, \ldots, N$:

1. Evaluate $x_{ij}$ using LLM  
2. Assign community label(s) set $\{c_j^i\}$ based on $x_{ij}$ and threshold $\tau$  
3. If $x_{ij} \ge \tau$:  
    Assign $\{c_j^i\}$ (possibly multiple communities)

**(2) Intra-Community Connections (BA Component)**

For each community $j = 1, \ldots, K$:

1. Construct a complete graph $G_j$ with $m_0$ nodes (i.e., every pair in $G_j$ is connected)  
2. Let $E_j$ be the set of edges in $G_j$  
3. Initialize node set:  
   $V_j \gets \{v_{j1}, v_{j2}, \dots, v_{jm_0}\}$

** (3) Iterative Construction**

For $i = m_0 + 1$ to $N$:

1. For each community label(s) set $\{c_j^i\}$ that node $i$ belongs to:  
   - For each existing node $e$ in $V_j$:  
     - Compute social influence score:  
       $
       S_e = \frac{F_e}{\sum_{t \in V_j} F_t}
       $  
       where $F_e$ is the follower count of node $e$  
   - Select $m$ distinct nodes from $V_j$ based on the social influence scores $\{S_e\}$, forming the set $\mathcal{E}$  
   - For each selected node $e \in \mathcal{E}$:  
     - Add edge $(e, i)$ to $E_j$  
2. Add node $i$ to $V_j$

**Return:** The final network $G(V, E)$

---

**Executation Script**: `python construct_network.py`

## 🥙 Add MBot/LBot to Dissemination Network

- The network built above is generated based on ordinary user agents. 

- Next, we introduce **malicious bots and legal bots** to simulate their impact in the process of dissemination and correction of false information.

- In order to ensure the controllability and scalability of the experiment, we control the proportion of the number of bots and the number of ordinary users connected to each bot through hyperparameter settings.

**Executation Script**: Execution `python Generate_Mbot.py` can generate human-like attributes for malicious bots and you can modulate the parameters in the `config.yaml` to control the number of bots.

**Executation Script**: Execution `python Concat_Human_Bots.py` can concatenate the human and bot attributes.

**Executation Script**: Execution `python construct_network.py` can generate the dissemination network including bots and humans.