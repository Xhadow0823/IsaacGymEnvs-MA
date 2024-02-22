import torch


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
