import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
import torch.utils.data.distributed
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
  def __init__(self):
    super(ToyModel, self).__init__()
    self.net1 = nn.Linear(10, 10)
    self.relu = nn.ReLU()
    self.net2 = nn.Linear(10, 5)

  def forward(self, x):
    x = self.net1(x)
    x = self.relu(x)
    x = self.net2(x)
    return x


class ToyDataset(torch.utils.data.Dataset):
  def __init__(self, N, feat, classes: int = 5):
    self.data = torch.randn(N, feat)
    self.labels = torch.randn(N, classes)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]

  def collate_fn(self, batch):
    data, labels = zip(*batch)
    return torch.stack(data), torch.stack(labels)


def demo_basic():
  torch.manual_seed(123)
  dist.init_process_group(backend="gloo")
  rank = dist.get_rank()
  print(f"Start running basic DDP example on rank {rank}.")

  # create model and move it to GPU with id rank
  model = ToyModel()
  ddp_model = DDP(model)

  N, feat = 20000, 10
  epochs = 5

  train_dataset = ToyDataset(N, feat)
  distributed_sampler = torch.utils.data.distributed.DistributedSampler(
      train_dataset,
      shuffle=True
  )
  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=2,
      sampler=distributed_sampler,
      collate_fn=train_dataset.collate_fn)

  loss_fn = nn.MSELoss()
  optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

  for epoch in range(epochs):
    for step, (sample_data, labels) in enumerate(
            tqdm(train_loader, disable=(rank != 0))):
      optimizer.zero_grad()
      outputs = ddp_model(sample_data)
      loss = loss_fn(outputs, labels)
      loss.backward()
      optimizer.step()

  dist.destroy_process_group()


if __name__ == "__main__":
  demo_basic()
