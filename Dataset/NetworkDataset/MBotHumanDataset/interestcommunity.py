# 可视化MBotHumanEdgeIndex的cohesion, 热图社群内的连接边数，社群间的连接边数

import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15  # 基础字体大小
communitylist = ["Business", "Education", "Entertainment", "Politics", "Sports", "Technology"]

# 加载不同社区的节点
path ="Dataset/CorrectDataset/NetworkDataset/MBotHumanDataset/HumanMBotData.json"
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 获取不同社区的节点
community_nodes = {}
for community in communitylist:
    community_nodes[community] = []
    for key in data.keys():
        user = data[key]
        if user['label'] == 'Human':
            interest_community = user['interest_community']
            if community in interest_community:
                community_nodes[community].append(user['id'])

# 加载edge_index
edge_index = np.load("Dataset/CorrectDataset/NetworkDataset/EdgeIndex/mbot_human_lbot_edge_index.npy").T.tolist()

# # 获取不同社区的连接边数

# community_edge_num_in = {}
# for community in communitylist:
#     community_edge_num_in[community] = 0
#     for edge in edge_index:
#         if edge[0] in community_nodes[community] and edge[1] in community_nodes[community]:
#             community_edge_num_in[community] += 1

# 获取社区间的连接边数, Business
BusinessList = ["Business", "Education", "Entertainment", "Politics", "Sports", "Technology"]
Business_edge_num_between = {}
for community in BusinessList:
    Business_edge_num_between[community] = 0
    for edge in edge_index:
        if edge[0] in community_nodes['Business'] and edge[1] in community_nodes[community]:
            Business_edge_num_between[community] += 1
        elif edge[1] in community_nodes['Business'] and edge[0] in community_nodes[community]:
            Business_edge_num_between[community] += 1

# 获取社区间的连接边数, Education
EducationList = ["Business", "Education", "Entertainment", "Politics", "Sports", "Technology"]
Education_edge_num_between = {}
for community in EducationList:
    Education_edge_num_between[community] = 0
    for edge in edge_index:
        if edge[0] in community_nodes['Education'] and edge[1] in community_nodes[community]:
            Education_edge_num_between[community] += 1  
        elif edge[1] in community_nodes['Education'] and edge[0] in community_nodes[community]:
            Education_edge_num_between[community] += 1

# 获取社区间的连接边数, Entertainment
EntertainmentList = ["Business", "Education", "Entertainment", "Politics", "Sports", "Technology"]
Entertainment_edge_num_between = {}
for community in EntertainmentList:
    Entertainment_edge_num_between[community] = 0
    for edge in edge_index:
        if edge[0] in community_nodes['Entertainment'] and edge[1] in community_nodes[community]:
            Entertainment_edge_num_between[community] += 1
        elif edge[1] in community_nodes['Entertainment'] and edge[0] in community_nodes[community]:
            Entertainment_edge_num_between[community] += 1

# 获取社区间的连接边数, Politics
PoliticsList = ["Business", "Education", "Entertainment", "Politics", "Sports", "Technology"]
Politics_edge_num_between = {}  
for community in PoliticsList:
    Politics_edge_num_between[community] = 0
    for edge in edge_index:
        if edge[0] in community_nodes['Politics'] and edge[1] in community_nodes[community]:
            Politics_edge_num_between[community] += 1
        elif edge[1] in community_nodes['Politics'] and edge[0] in community_nodes[community]:
            Politics_edge_num_between[community] += 1    

# 获取社区间的连接边数, Sports
SportsList = ["Business", "Education", "Entertainment", "Politics", "Sports", "Technology"]
Sports_edge_num_between = {}
for community in SportsList:
    Sports_edge_num_between[community] = 0
    for edge in edge_index:
        if edge[0] in community_nodes['Sports'] and edge[1] in community_nodes[community]:
            Sports_edge_num_between[community] += 1 
        elif edge[1] in community_nodes['Sports'] and edge[0] in community_nodes[community]:
            Sports_edge_num_between[community] += 1

# 获取社区间的连接边数, Technology
TechnologyList = ["Business", "Education", "Entertainment", "Politics", "Sports", "Technology"]
Technology_edge_num_between = {}
for community in TechnologyList:
    Technology_edge_num_between[community] = 0      
    for edge in edge_index:
        if edge[0] in community_nodes['Technology'] and edge[1] in community_nodes[community]:
            Technology_edge_num_between[community] += 1
        elif edge[1] in community_nodes['Technology'] and edge[0] in community_nodes[community]:
            Technology_edge_num_between[community] += 1

# 将上述六个字典合并成一个6*6的矩阵
cohesion_matrix = np.zeros((6, 6))
for i, community in enumerate(communitylist):
    cohesion_matrix[i, 0] = Business_edge_num_between[community]
    cohesion_matrix[i, 1] = Education_edge_num_between[community]
    cohesion_matrix[i, 2] = Entertainment_edge_num_between[community]
    cohesion_matrix[i, 3] = Politics_edge_num_between[community]
    cohesion_matrix[i, 4] = Sports_edge_num_between[community]
    cohesion_matrix[i, 5] = Technology_edge_num_between[community]

# 生成下三角掩码（mask上三角）
mask = np.triu(np.ones_like(cohesion_matrix, dtype=bool), k=1)
# 创建热图
plt.figure(figsize=(8, 6))
# 定义分类标签
# categories = ["Business", "Education", "Entertainment", "Politics", "Sports", "Technology"]
categories = ["Bu", "Ed", "En", "Po", "Sp", "Te"]

heatmap = sns.heatmap(
    cohesion_matrix,
    annot=True,          # 显示数值
    fmt=".0f",
    cmap="Blues",       # 颜色映射（黄-橙-红）
    mask=mask,                # 屏蔽上三角
    vmin=0,              # 颜色条最小值
    vmax=450,              # 颜色条最大值
    linewidths=0.5,      # 单元格边框线宽
    linecolor="white",   # 单元格边框颜色
    square=True,          # 保持单元格为正方形
    xticklabels=categories,
    yticklabels=categories,
    annot_kws={"size": 25}  # 设置注释字体大小
)
# 设置颜色条字体大小
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=25)

# 旋转X轴标签以便更好地显示
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=25)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=25)
# 调整布局防止标签被截断
plt.tight_layout()
# 保存热图
plt.savefig("Dataset/CorrectDataset/NetworkDataset/MBotHumanDataset/interest_community.png", dpi=300, bbox_inches='tight')   