import torch
import torch.distributed as dist
import torch.nn as nn
from pathlib import Path
import os
def parallelize(rank, config, model, find_unused_parameters=True):
    if config['distributed']:
        # Use DDP
        config['multigpu'] = True
        gpu = rank + config['device'] if config['device_ids'] == None else config['device_ids'][rank]
        
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        dist.init_process_group(backend="nccl", world_size=config['world_size'], rank=rank)
        config['batch_size'] = int(config['batch_size'] / config['world_size'])

        config['num_workers'] = int(
            (config['num_workers'] + config['world_size'] - 1) / config['world_size'])
        
        print("Device", gpu, "Rank",  rank, "batch size",
              config['batch_size'], "Workers", config['num_workers'])
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(rank).float()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank,
                                                          find_unused_parameters=find_unused_parameters)

    elif config['world_size'] == 1:
        model = model.to(rank)
        config['multigpu'] = False

    elif config['world_size'] > 1:
        # Use DP
        config['multigpu'] = True
        model = model.to(config['device']).float()

        if config['device_ids'] is None:
            # If no device IDs are provided, create a list starting from the base device.
            base_device = config['device']
            world_size = config['world_size']
            device_ids = [base_device + i for i in range(world_size)]
        else:
            device_ids = config['device_ids']

        # Wrap the model with DataParallel to enable multi-GPU training.
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    return model
