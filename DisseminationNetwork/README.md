<div align="center">
  <img src="/Users/boyuqiao/Desktop/BotInfluence-main/BotInfluence/DisseminationNetwork/Figure/GPT4oGeneratedFigure.png" alt="Construct Disinformation Netwerk" height="375">
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
- User interest community scores: $\mathcal{IC}_{uj}$  
- Predefined threshold for community assignment: $\tau$  
- Social influence scores for nodes: $\mathcal{SI}_u$  
- Number of edges each new node forms: $m \ (m \le m_0)$  
- Community label(s) of each user $i$: $\{C_j^i\}$

**Outputs:**

- Graph $\mathcal{G}(\mathcal{V}, \mathcal{E})$ representing the constructed BA-SBM network

**(1) Node Community Assignment (SBM Component)**

For each initial node $u = 1, \ldots, N$:

1. Evaluate $\mathcal{IC}_{uj}$ using LLM  
2. Assign community label(s) set $\{C_j^u\}$ based on $\mathcal{SI}_{ij}$ and threshold $\tau$  
3. If $\mathcal{SI}_{ij} \ge \tau$:  
    Assign $\{C_j^i\}$ (possibly multiple communities)

**(2) Intra-Community Connections (BA Component)**

For each community $j = 1, \ldots, K$:

1. Construct a complete graph $G_j$ with $m_0$ nodes (i.e., every pair in $G_j$ is connected)  
2. Let $E_j$ be the set of edges in $G_j$  
3. Initialize node set:  
   $V_j \gets \{v_{j1}, v_{j2}, \dots, v_{jm_0}\}$

**(3) Iterative Construction**

For $v = m_0 + 1$ to $N$:

1. For each community label(s) set $\{C_j^v\}$ that node $v$ belongs to:  
   - For each existing node $e$ in $V_j$:  
     - Compute social influence score:  
       $
       \mathcal{SI}_e = \frac{F_e}{\sum_{t \in V_j} F_t}
       $  
       where $F_e$ is the follower count of node $e$  
   - Select $m$ distinct nodes from $V_j$ based on the social influence scores $\{S_e\}$, forming the set $\mathcal{E}$  
   - For each selected node $e \in \mathcal{E}$:  
     - Add edge $(e, i)$ to $E_j$  
2. Add node $i$ to $V_j$

**Return:** The final network $\mathcal{G}(\mathcal{V}, \mathcal{E})$

---

**Executation Script**: `python run.py`

## 🥙 Add MBot/LBot to Dissemination Network

- The network built above is generated based on ordinary user agents. 

- Next, we introduce **malicious bots and legal bots** to simulate their impact in the process of dissemination and correction of false information.

- In order to ensure the controllability and scalability of the experiment, we control the proportion of the number of bots and the number of ordinary users connected to each bot through hyperparameter settings.

**Executation Script**: Execution `python Generate_Mbot.py` and `python generate_lbot.py` can generate human-like attributes for malicious and legitimate bots and you can modulate the parameters in the `config.yaml` to control the number of M/Lbots.

**Executation Script**: Execution `python Concat_MBot_Human_LBot.py` can concatenate the human and bot attributes.

**Executation Script**: Execution `python GenerateMLNetwork.py` can generate the dissemination network including bots and humans.