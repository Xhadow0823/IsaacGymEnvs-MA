import torch



original = torch.zeros(3, 3)
partial  = original[:, 1]

print(original)
print(partial)

partial[:] = torch.Tensor([1, 2, 3])

print(original)
