import argparse
from pathlib import Path
import yaml
from torch.utils.data.dataloader import DataLoader
import logging
from misc.distribute import parallelize
import torch.multiprocessing as mp
import torch
import os
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torch.utils.data._utils.collate import default_collate
from PIL import Image

def fix_random_seed(seed: int):
    import random
    import numpy

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_model_path(config):
    if config['model_name'] == 'bart-large':
        model_path = 'facebook/bart-large-mnli'
    elif config['model_name'] == 'roberta-large':
        model_path = 'FacebookAI/roberta-large-mnli'
    elif config['model_name'] == 'bert':
        model_path = 'bert-base-uncased'
    elif config['model_name'] == 'roberta-base':
        model_path = 'WillHeld/roberta-base-mnli'
    else:
        raise ValueError("Invalid model name.")
    return model_path

def get_emoji_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
    return tokenizer
def get_text_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer

def custom_collate_fn(batch):
    batch_out = {}
    text = [item["EN"] for item in batch]
    text_tokenizer = get_text_tokenizer()
    batch_out["text_tokens"] = text_tokenizer(text, padding=True, return_tensors="pt")  # Pad dynamically
    batch_out["labels"] = torch.stack([item["labels"] for item in batch])
    emojis = [item["emojis"] for item in batch]
    emoji_tokenizer = get_emoji_tokenizer()
    batch_out["emoji_tokens"] = emoji_tokenizer(emojis, padding=True, return_tensors="pt")  # Pad dynamically
    batch_out["images"] = [item["images"] for item in batch]
    return batch_out




def build_model(config):
    from model import FinalModel
    if config['half_precision']:
        weight_dtype = torch.float16
        logging.info(f"Running with half precision ({weight_dtype}).")
    else:
        weight_dtype = torch.float32
    model = FinalModel(device=config['device'])
    model.to(config['device'])
    return model

def get_dataloader(config):
    from torchvision.transforms import Compose
    import torch
    from torch.utils.data import DataLoader
    from dataset.dataset import EmoteDataset, DatasetMode

    if config['seed'] is None:
        loader_generator = None
    else:
        loader_generator = torch.Generator().manual_seed(config['seed'])


    # Create the training dataset
    train_config = config['train_dataset']
    train_dataset = EmoteDataset(
        csv_file=train_config['csv_file'],
        dataset_dir=config['dataset_dir'],
        portion=config.get('portion', 1.0),
        random_state=config['seed'],
        mode=DatasetMode.TRAIN
    )

    val_config = config['val_dataset']
    val_dataloader = None
    # Create the validation dataset
    val_dataset = EmoteDataset(
        csv_file=val_config['csv_file'],
        dataset_dir=config['dataset_dir'],
        portion=1.0,  # use full data for validation
        random_state=config['seed'],
        mode=DatasetMode.EVAL
    )

    # Create the DataLoader for training
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        collate_fn=custom_collate_fn,
        generator=loader_generator,
        num_workers=config.get('num_workers', 4)
    )

    # Create the DataLoader for validation
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=config.get('num_workers', 4)
    )
    
    return train_dataloader, val_dataloader


def main_worker(rank: int, config: dict):
    """
    Main worker function for training or validation on a given process (rank).

    Args:
        rank (int): The rank of the current process.
        config (dict): Configuration dictionary containing training parameters.
    """
    from trainer.EmoteTrainer import EmoteTrainer
    import torch.distributed as dist
    import traceback
    import wandb
    from transformers import AutoTokenizer

    try:
        # Ensure reproducibility.
        fix_random_seed(config['seed'])
        # model_path = get_model_path(config)
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        # tokenizer.add_tokens(['[EM]'])
        os.environ['TOKENIZERS_PARALLELISM']='true'
        # Build and parallelize the model.
        model = build_model(config)
        config.update({'rank': rank})
        model = parallelize(rank, config, model)

        # (Optional) Uncomment below to log the total number of parameters.
        # total_params = round(count_parameters(model) / 1e6, 2)
        # config['total_params'] = f"{total_params}M"
        # print(f"Total parameters: {total_params}M")

        # Obtain dataloaders for training and validation.
        train_dataloader, val_dataloader = get_dataloader(config)

        # Instantiate the trainer.
        trainer = EmoteTrainer(
            config, model, train_dataloader, device=rank
        )

        # Train or validate based on configuration.
        if config['train']:
            trainer.train()
        else:
            metric, loss = trainer.validate()
            if not config['distributed'] or config.get('rank', 0) == 0:
                print(metric, loss)

    finally:
        # clean up the distributed process group and finish wandb logging.
        # save checkpoint in case of failure
        trainer.save_checkpoint("corrupt.pth")
        dist.barrier()
        dist.destroy_process_group()
        wandb.finish()

if __name__ == "__main__":
     
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config', type=Path, help='Path to config.yaml', default='./config/trainer_config.yaml')
    parser.add_argument('--specific_config', type=Path, help='Path to config.yaml', default='./config/specific_config.yaml')
    args = parser.parse_args()

    if not (args.train_config.exists() and args.specific_config.exists()):
        print(f"Config file \"{args.train_config}\" or \"{args.specific_config}\" does not exist!\n")
        exit()

    with args.train_config.open() as file:
        config = yaml.safe_load(file)

    with args.specific_config.open() as file:
        config.update(yaml.safe_load(file))

    # Overwrite config.yaml settings with command line settings
    for arg in vars(args):
        if getattr(args, arg) is not None and arg in config:
            config[arg] = getattr(args, arg)
    

    os.environ['CUDA_VISIBLE_DEVICES'] = "".join(str(x)+',' for x in config['device_ids'])
    
    if config['distributed']:
        mp.spawn(main_worker, nprocs=config['world_size'], args=(config,))
    else:
        rank = config['device'] if config['world_size'] == 1 else 0
        main_worker(rank, config)