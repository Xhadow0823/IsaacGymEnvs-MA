import torch
from torch import Tensor


A = torch.arange(3*6*7).view(3, 6, 7)
B = torch.arange(3*6*7).view(3, 6, 7) + (3*6*7)

print(A)
print(B)

print(A.shape)
print(B.shape)

C = torch.stack([A, B], dim=1).view(3*2, 6, 7)
print(C)
print(C.shape)


exit()


num_agents = 2
B = torch.arange(2 * 3).view(2, 3).repeat_interleave(num_agents, dim=0)
print(B)

exit()


A = torch.arange(6).view(2, 3)
print(A)
B = torch.eye(2, 3).to(dtype=torch.long)
print(B)

print(A * B)  # pair-wise mul

print(A.T @ B)  # matrix mul

exit()


dofs = torch.arange(2*35).view(2, 35)
print(dofs)

# 13 32 ...
site = dofs[:, 13::19]
print(site)

site[...] = 99
print(dofs)

exit()

dofs = torch.arange(2*18).view(2, 18)
print(dofs)
# 0:0+7 9:9+7 -> 改為9個一組，選前7個
arm_dofs = dofs.view(-1, 9)[:, :7]
print(arm_dofs)

arm_dofs[...] = 99
print(dofs)

exit()

a = torch.arange(6).view(2, 3)
print(a)

b = a.unsqueeze(1).repeat_interleave(2, dim=1)
print(b)

exit()

a = torch.ones((2, 3))

a = a * 9

print(a)

b = torch.ones(2, 2, 3)

b[0, ...] = 5
b[1, ...] = 3


c = a - b
print(c)


exit()

# 假設 A 是一個 2x18 的 tensor
A = torch.randn(2, 18)

# 建立索引列表以選取指定的列
indices = torch.cat((torch.arange(7), torch.arange(9, 16)))

# 使用 torch.index_select() 從 A 中選取指定的列並建立 B
B = torch.index_select(A, dim=1, index=indices)

# 確認 B 是否與 A 共享相同的資料
B[0, 0] = 999  # 修改 B 中的值
print(A[0, 0])  # 檢查 A 中對應位置的值是否也被修改

exit()

num_envs = 2
num_dofs = 18

dofs = torch.arange(num_envs * num_dofs).view(num_envs, -1)
print(dofs)

t1 = dofs[:, [*range(7)]]
print(t1)
# t1[:, :] = 99

t2 = dofs[:, :7]  # NOTE: 使用 : 或是 ... 才是原本的tensor 的 partial view，用 list 作為 indice 僅能讀取，不能修改原本的資料！！
print(t2)
t2[:, :] = 99

print(dofs)




# dofs[[*range(7), *range(9,9+7)]] = 99
# print(dofs)

exit()

num_agents = 2

# agent_ids = torch.arange(8).view(-1, num_agents)
agent_ids = torch.tensor([0,  6])
print(agent_ids)

def from_agent_to_env(agent_ids: Tensor):
    global num_agents
    return agent_ids // num_agents

env_ids = from_agent_to_env(agent_ids)
print(env_ids)

# env OR filter
env_or_ids = env_ids.unique()
print(env_or_ids)

# env AND filter
env_and_ids = env_ids.bincount()
env_and_ids = torch.arange(env_and_ids.numel())[env_and_ids >= num_agents]
print(env_and_ids)

exit()

# # env_ids = torch.arange(3).tolist()
# env_ids = torch.tensor([0, 6])
# print(env_ids)

# num_agents = 2
# new_env_ids = torch.tensor( [list(range(2*i, 2*i+num_agents)) for i in env_ids] )
# print(new_env_ids)

# exit()

env_ids = torch.arange(20).view(-1, 5)
print(env_ids)
new_env_ids = env_ids[:, [0, 4]]
print(new_env_ids)

new_new_env_ids = new_env_ids[[0, 1], :]
print(new_new_env_ids)

env_ids2 = env_ids[[0, 1], :][:, [0, 4]]
print(env_ids2)

exit()

env_ids = torch.arange(4).view(2, 2)
print(env_ids)

new_env_ids = env_ids.repeat_interleave(2, dim=0)
print(new_env_ids)

exit()

env_ids = torch.arange(4).view(4, -1)
print(env_ids)
temp = torch.cat([env_ids, env_ids], dim=-1)
print(temp)
agent_to_env_ids = temp.flatten()
print(agent_to_env_ids)
print(env_ids.repeat_interleave(2))  # this is good

agent_ids = torch.zeros(8, dtype=torch.int32)
print(agent_ids)

exit()

original = torch.zeros(3, 3)
partial  = original[:, 1]

print(original)
print(partial)

partial[:] = torch.Tensor([1, 2, 3])

print(original)
