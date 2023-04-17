import torch
from torch.utils.data import DataLoader

class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.arange(100).view(100, 1).float()
        
    def __getitem__(self, index):
        print(index)
        x = self.data[index]
        return x
    
    def __len__(self):
        return len(self.data)

dataset = MyDataset()
print(len(dataset))
"""        
sampler = torch.utils.data.sampler.BatchSampler(
    torch.utils.data.sampler.RandomSampler(dataset),
    batch_size=4,
    drop_last=False)
"""
sample_weights = 100*[1.]
# E.g. if we want a sample to show up on average in every batch, 
weight_important = 0.25
weight_other = (1.0 - weight_important)/100
sample_weights = 100*[weight_other]
sample_weights[0] = weight_important
#sample_weights[1] = weight_important
# NOTE: If you want more samples to show up on average in every batch, need to 
#       probably increase the batch size, otherwise you'll have batches full
#       of the important samples only.
# NOTE To remedy above, simply set "important" samples with weight 1/batch_size.
#      This will ensure that such samples get seen once per batch, on average.
#      Then, one must care to set the batch size large enough so that the entire 
#      batch isn't full of such samples. E.g. probably beneficial to go above 
#      batch_size = 4 if we have ~2-3 important samples.

num_samples = 100
sampler = torch.utils.data.sampler.BatchSampler(
    torch.utils.data.WeightedRandomSampler(weights=sample_weights,
                                           num_samples=num_samples,
                                           replacement=True),
    batch_size=4,
    drop_last=False)

"""
loader = DataLoader(
    dataset,
    sampler=sampler)
"""

loader = DataLoader(
    dataset,
    batch_sampler=sampler)

count = 0
for data in loader:
    print("New data:")
    print(data)
    count += 1

print(f"Number of batches looped over: {count}")