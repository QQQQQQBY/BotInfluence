# Agent Attributes

<p align="center">
  <img src="/Users/boyuqiao/Desktop/BotInfluence-main/BotInfluence/UserBotAgentAttributes/Figure/GPT4oGeneratedFigure.png" 
       alt="Agent Attributes"
       width="60%" 
       style="border: 1px solid #eee; box-shadow: 2px 2px 5px rgba(0,0,0,0.1)">
</p>
</p>

<p align="center">
  <em>🤖  The necessary attributes required for simulating the dissemination of false information 🤖 </em>
</p>

## 🛝 Dataset Prepare

The Dataset folder contains the user behavior dataset we collect and process from the X/Twitter platform, including:

- 🌍 **Data Source**: General user data collected from six specific communities actively participating in discussions (Entertainment, Technology, Sports, Business, Politics, and Education.)

- 🏛️ **Data Content**: User basic information file and the last 200 posts per user.

🌷 **Data Format**:

```
{
        "id": 1,
        "user_id": 1250830691824283648,
        "user_name": "Evan",
        "field": "Business",
        "profile": {
            "introduction": "Free Stock Market News that is FAST, ACCURATE, CONSISTENT, and RELIABLE | Not Just Stock News | My Daily Stock Market Recap is the link in my bio ⬇️",
            "is_alive": 1,
            "is_protected": 0,
            "is_verified": 1,
            "tweets_count": 110513,
            "followers_count": 537311,
            "following_count": 378,
            "create_time": "2020-04-16 16:57:54",
            "insert_time": "2025-02-10 16:36:58"
        },
        Posts:{"1": {
                "message_id": "1889715380559470998",
                "create_time": "2025-02-12 16:37:10",
                "content": "AKO CAPITAL JUST UPDATED ITS $7.2 BILLION PORTFOLIO https://t.co/yCaqWg48Ps",
                "comment_count": 6,
                "share_count": 1,
                "like_count": 40,
                "retweet_count": 3,
                "read_count": 9850,
                "language": "en",
                "field": "Business"
            },
            ...
            },
        Retweet:{"1": {
                "message_id": "f65699625f47fabd8e497892f6a5506e",
                "create_time": "2025-02-12 15:47:09",
                "content": "RT @meetblossomapp:  Inflation every month since 2015 (CPI MoM) https://t.co/voPKjp2JrP",
                "comment_count": 0,
                "share_count": 0,
                "like_count": 0,
                "retweet_count": 16,
                "read_count": 0,
                "language": "en",
                "field": "Business"
            },
            ...
            },
        Quote:{"1": {
                "message_id": "1889699497090945072",
                "create_time": "2025-02-12 15:34:03",
                "content": "I will be doing some in-depth market research about Apple getting into Humanoid Robots and will report back  https://t.co/OWdlcWboOR",
                "comment_count": 10,
                "share_count": 0,
                "like_count": 60,
                "retweet_count": 2,
                "read_count": 21301,
                "language": "en",
                "field": "Business"
            },
            ...
            }
```

- 🏗️ **Data Processing** : Time truncation process has been performed (see Appendix 1.1 for details). Complete data cleaning and standardization

- 🎡 **Data Acquisition**: The original dataset is archived on [Google Drive](https://drive.google.com/drive/folders/1vhbEEu0HJvlHmxHYM9rVWXmBstAQFrvr?usp=sharing). This folder provides preprocessed versions that can be used directly for analysis. 


## 👨‍🚀 User/Bot Agent Attributes

The contents in this folder correspond to the **user attribute modeling** part described in our paper Section 3.1.1 of this article, mainly including the user attribute parameters required to construct a disinformation propagation simulation:

One-click execution for handling user attribute information: `python Concat_Five_Attribute.py`

 1️⃣ **Interest Communities**: Distribution of user interest communities (fields)
 
 2️⃣ **Trust Threshold**: User trust threshold for information in different fields

 3️⃣ **Dissemination Tendency**: Probability of users' tendency to share false information in various communities

 4️⃣ **Social Influence**: User social influence indicators

 5️⃣ **Activation Time**: User activation probability in time step simulation

### 🍭 Interest Communities 

Although the data was collected from X/Twitter’s Community module, considering that users usually have cross-community interests, we selected 6 typical communities as the sampling scope to ensure that the research objects are all affiliated with these target communities. In order to accurately identify the user's multi-community interest distribution, we developed an interest assessment model based on user historical behavior data. Specific implementations include:


🚄 **Prompt Engineering**: Building dynamic assessments using prompt engineering

🚊 **Prompt Template**: path: `Prompts/InterestCommunity.yaml`

🚗 **Automated Evaluation**: Execute: `python interest_community.py`


### 🦄 Trust Threshold 

There are significant differences in the ability of users to identify false information in different fields. In order to accurately simulate the dynamics of false information dissemination, we used the prompt engineering to evaluate the ability of users to identify information in different communities.

🚄 **Prompt Engineering**: Building dynamic assessments using prompt engineering

🚊 **Prompt Template**: path: `Prompts/TrustThreshold.yaml`

🚗 **Automated Evaluation**: Execute: `python trust_threshold.py`

### ⛺ Dissemination Tendency

The propagation behavior of users presents a significant **power law distribution** feature, that is, **a few active users contribute most of the propagation volume, while the propagation behavior of most users is relatively limited**. To accurately describe this phenomenon, we use exponentially truncated power law distribution for modeling based on the number of information sharing times (including forwarding and citation behaviors) by the user within a fixed time window. 

Further, considering the differences in user communication preferences in different communities, we innovatively propose a composite model: this model combines **truncated power law distribution** with probability valuations based on **community interests** to jointly characterize the user's communication tendency. The specific modeling formula is as follows:

$$
  DT_{ij} = \theta \cdot (C\cdot (x_i)^{-\alpha} + |\epsilon(x_i)|) + (1-\theta) \frac{IC_{ij}}{ {\textstyle \max_{j} IC_{ij}}}, \  x_i \ge x_{min},
$$

🎡 **Automated Calculation**: Execute: `python disseminate_tendency.py`

### 🦋 Social Influence

In the built user information dissemination simulation network, the influence of users shows significant differences: **some high-influence users can promote the widespread dissemination of information they share, while the information of low-influence users often has a limited reach**. 

This dissemination model is highly similar to the influence distribution based on the number of fans in real social networks. Therefore, we use the number of **users' followers** as a key indicator to quantify their social influence. The specific calculation formula is as follows:

$w_i = \frac{f_i}{\sum_{j=1}^{n} f_j}$

$f_i$ denotes user $i$'s follower number

🎡 **Automated Calculation**: Execute: `python social_influence.py`


### 🐌 Activation Time

In real social networks, users exhibit different active time patterns. To quantify this feature, we **calculate the frequency of time active for the user within a fixed observation window**, i.e. **the ratio of the number of actives for a particular user at a given time step to the total number of actives**.

Among them, we define the user's active state (activated) as the occurrence of any of the following behaviors:

- Publish original content

- Forward other people's information

- Quote other content

🎡 **Automated Calculation**: Execute: `python activation_time.py`

## 🐞 Summarize the Above Attribute Information
Run the attribute merge script: `python Concat_Five_Attribute.py`

This script will integrate all user attribute characteristics described above.

The merged complete attribute dataset is stored in the path: `Dataset/AttributeDataset/concat_attribute_data.json`

🌸 **Format**:
```
{"1878730280472633345": {
        "id": 0,
        "user_id": "1878730280472633345",
        "community": {
            "Entertainment": 1,
            "Technology": 6.0,
            "Sports": 5.0,
            "Business": 1,
            "Politics": 8.0,
            "Education": 1
        },
        "trust_threshold": {
            "Entertainment": 0.6,
            "Technology": 0.75,
            "Sports": 0.5,
            "Business": 0.8,
            "Politics": 0.65,
            "Education": 0.55
        },
        "dissemination_tendency": {
            "Entertainment": 0.7945287761936065,
            "Technology": 0.4445287761936065,
            "Sports": 0.4445287761936065,
            "Business": 0.4445287761936065,
            "Education": 0.4445287761936065,
            "Politics": 0.5445287761936065
        },
        "social_influence": 649,
        "activation_time": [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ],
        "sharetimes": 0
    },
    ...
} 
```

