import math
import random
import torch
from torch.utils.data import Sampler

class DistributedBucketSampler(Sampler):
    """ Bucketing by length, then distributed sampling。
    
    Args:
        lengths (List[int]): the list of length of each sample
        batch_size (int): batch size for each progress
        num_replicas (int): world_size
        rank (int): local rank
        shuffle (bool): whether to shuffle samples in each bucket
        drop_last (bool): whether to drop out the last samples less than batch_size
    """
    def __init__(self,
                 lengths,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 drop_last=False,
                 seed = 0):
        
        if num_replicas is None:
            if not torch.distributed.is_initialized():
                raise RuntimeError("Need 'num_replicas' or initialize torch.distributed")
            num_replicas = torch.distributed.get_world_size()

        if rank is None:
            if not torch.distributed.is_initialized():
                raise RuntimeError("Need 'rank' or initialize torch.distributed")
            rank = torch.distributed.get_rank()

        self.lengths = lengths
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last

        # all samples
        self.num_samples = len(self.lengths)

        # all batch number of each epoch in each rank
        # 如果不丢弃多余的桶，那么要向上取整
        total_batches = math.ceil(self.num_samples / (self.batch_size * self.num_replicas))
        self.num_batches_per_replica = total_batches
        self.seed = seed
        self.epoch = 0
    
    def set_epoch(self, epoch: int):
        """ control and confirm the consistence of shuflle for each progress in DDP """
        self.epoch = epoch

    def __iter__(self):
        # set random seed
        if self.shuffle:
            shuffler = random.Random(self.seed + self.epoch)
            
        # 1. sort by length
        sorted_indices = list(range(self.num_samples))
        sorted_indices.sort(key=lambda idx: self.lengths[idx])

        # 2. divided indices to buckets, each is batch_size * num_replicas
        bucket_size = self.batch_size * self.num_replicas
        buckets = [
            sorted_indices[i: i + bucket_size]
            for i in range(0, len(sorted_indices), bucket_size)
        ]
        if self.drop_last and len(buckets[-1]) < bucket_size:
            buckets = buckets[:-1]

        # 3. shuffle samples for each bucket
        if self.shuffle:
            for b in buckets:
                shuffler.shuffle(b)

        # 4. divide the bucket to some mini-batches, distribute to each rank
        all_batches = []
        for bucket in buckets:
            # bucket 可能大小不整除，按照 batch_size 切分
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        # 5. distribute indices to multiple GPUs
        selected = [
            batch for i, batch in enumerate(all_batches)
            if (i % self.num_replicas) == self.rank
        ]

        if self.shuffle:
            shuffler.shuffle(selected)

        # return indices
        return iter(selected)

    def __len__(self):
        return self.num_batches_per_replica
