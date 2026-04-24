import math
import random
import torch
from torch.utils.data import Sampler
from collections import Counter

class DistributedBucketSampler(Sampler):
    """ Bucketing by length, then distributed sampling.
    
    Args:
        lengths (List[int]): the list of length of each sample
        batch_size (int): batch size for each progress
        num_replicas (int): world_size
        rank (int): local rank
        shuffle (bool): whether to shuffle samples in each bucket
        drop_last (bool): whether to drop out the last samples less than batch_size
        seed (int): random seed
        cell_types (List[str]): List of cell type strings corresponding to each sample, used for balanced sampling.
        balance_classes (bool): Whether to enable class balancing (automatically downsamples majority classes).
    """
    def __init__(self,
                 lengths,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 drop_last=False,
                 seed=0,
                 cell_types=None,
                 balance_classes=False):   
        
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
        self.seed = seed
        self.epoch = 0
        
        # Balanced sampling initialization logic
        self.balance_classes = balance_classes
        if self.balance_classes:
            if cell_types is None:
                raise ValueError("cell_types must be provided when balance_classes=True")
            
            # Count occurrences of each cell type string
            counts = Counter(cell_types)
            self.min_count = min(counts.values())
            self.classes = list(counts.keys())
            
            # Group indices by class for dynamic sampling in __iter__
            self.class_to_indices = {c: [] for c in self.classes}
            for idx, c in enumerate(cell_types):
                self.class_to_indices[c].append(idx)
            
            # Calculate the theoretical total samples after balancing
            self.num_samples = self.min_count * len(self.classes)
            print(f"[Sampler] Balanced Sampling ON: Downsampling all classes to {self.min_count} samples. Total effective samples: {self.num_samples}")
        else:
            self.num_samples = len(self.lengths)

        # Strictly calculate the number of batches assigned to the current Rank
        bucket_size = self.batch_size * self.num_replicas
        if self.drop_last:
            self.num_batches_per_replica = math.floor(self.num_samples / bucket_size)
        else:
            self.num_batches_per_replica = math.ceil(self.num_samples / bucket_size)
            
        # Calculate the total required samples globally for subsequent padding to prevent DDP deadlocks
        self.total_size = self.num_batches_per_replica * bucket_size

    def set_epoch(self, epoch: int):
        """ control and confirm the consistence of shuflle for each progress in DDP """
        self.epoch = epoch

    def __iter__(self):
        # set random seed (needed if shuffle or balanced sampling is enabled)
        if self.shuffle or self.balance_classes:
            shuffler = random.Random(self.seed + self.epoch)
            
        # ---------------------------------------------------------
        # 1. Obtain sample indices participating in the training for this Epoch
        # ---------------------------------------------------------
        if self.balance_classes:
            selected_indices = []
            for c in self.classes:
                class_idx_list = self.class_to_indices[c].copy()
                # Shuffle intra-class indices using the epoch-aware shuffler
                # Ensuring different majority class samples are selected each Epoch!
                shuffler.shuffle(class_idx_list) 
                # Truncate to the minimum count to achieve Downsampling
                selected_indices.extend(class_idx_list[:self.min_count])

            # global shuffle
            if self.shuffle:
                shuffler.shuffle(selected_indices)
        else:
            selected_indices = list(range(len(self.lengths)))

            # global shuffle
            if self.shuffle:
                shuffler.shuffle(selected_indices)

        # ---------------------------------------------------------
        # 2. Padding / Truncating to ensure strict synchronization across DDP processes
        # ---------------------------------------------------------
        # In distributed training, non-divisible totals cause early exits and deadlocks on some GPUs
        if len(selected_indices) < self.total_size:
            # Padding: randomly draw from the beginning to fill the gap
            padding_size = self.total_size - len(selected_indices)
            selected_indices += selected_indices[:padding_size]
        elif len(selected_indices) > self.total_size:
            # Truncating (usually triggered when drop_last=True)
            selected_indices = selected_indices[:self.total_size]

        # ---------------------------------------------------------
        # 3. Bucket logic applied to the filtered selected_indices
        # ---------------------------------------------------------
        # 3.1 sort by length
        selected_indices.sort(key=lambda idx: self.lengths[idx])

        # 3.2 divided indices to buckets, each is batch_size * num_replicas
        bucket_size = self.batch_size * self.num_replicas
        buckets = [
            selected_indices[i: i + bucket_size]
            for i in range(0, len(selected_indices), bucket_size)
        ]
        
        # 3.3 shuffle samples for each bucket
        if self.shuffle:
            for b in buckets:
                shuffler.shuffle(b)

        # 3.4 divide the bucket to some mini-batches, distribute to each rank
        all_batches = []
        for bucket in buckets:
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                # Since padding/truncating is done, this technically isn't needed, but kept as a failsafe
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        # 3.5 distribute indices to multiple GPUs
        selected_batches = [
            batch for i, batch in enumerate(all_batches)
            if (i % self.num_replicas) == self.rank
        ]

        # Shuffle the order of Batches (ensures Batches of similar lengths don't always stay together)
        if self.shuffle:
            shuffler.shuffle(selected_batches)

        # return indices
        return iter(selected_batches)

    def __len__(self):
        return self.num_batches_per_replica