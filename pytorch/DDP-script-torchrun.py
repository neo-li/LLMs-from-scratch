import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import platform
# divide the dataset among these different processes

from torch.utils.data.distributed import DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP

# the init_process_group function should be called at the beginning of the training script to initialize a process group for each process in the distributed setup, 
# and destroy_process_group should be called at the end of the training script to destroy a given process group and release its resources.
from torch.distributed import init_process_group, destroy_process_group



'''
sets the master node’s address and communication port (unless already provided by torchrun), 
initializes the process group using the NCCL backend (which is optimized for GPU-to-GPU communication), 
and then sets the device for the current process using the provided rank.
'''
def ddp_setup(rank, world_size):
    """
    Arguments:
        rank: a unique process ID
        world_size: total number of processes in the group
    """
    # Only set MASTER_ADDR and MASTER_PORT if not already defined by torchrun
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"

    # initialize process group
    if platform.system() in(["Windows", "Darwin"]):  # Windows and macOS have different default backends
        # Disable libuv because PyTorch for Windows isn't built with support
        os.environ["USE_LIBUV"] = "0"
        # Windows users may have to use "gloo" instead of "nccl" as backend
        # gloo: Facebook Collective Communication Library
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl: NVIDIA Collective Communication Library
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # torch.cuda.set_device(rank) # don't have cuda but mps

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


def prepare_dataset():
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    # Uncomment these lines to increase the dataset size to run this script on up to 8 GPUs:
    # factor = 4
    # X_train = torch.cat([X_train + torch.randn_like(X_train) * 0.1 for _ in range(factor)])
    # y_train = y_train.repeat(factor)
    # X_test = torch.cat([X_test + torch.randn_like(X_test) * 0.1 for _ in range(factor)])
    # y_test = y_test.repeat(factor)

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,  # NEW: False because of DistributedSampler below
        pin_memory=False, # NEW: pin_memory is not needed when using CPU or MPS 
        drop_last=True,
        # NEW: chunk batches across GPUs without overlapping samples:
        sampler=DistributedSampler(train_ds)
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )
    return train_loader, test_loader


# NEW: wrapper
def main(rank, world_size, num_epochs):

    ddp_setup(rank, world_size)  # NEW: initialize process groups

    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)

    # we transfer both the model and data to the correct GPU using .to(rank), 
    # where rank corresponds to the GPU index for the current process.

    device = "cpu"
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    # model = DDP(model, device_ids=[rank])  # NEW: wrap model with DDP
    # (CPU DDP has no device_ids)

    # the core model is now accessible as model.module
    # enables synchronized gradient updates across all GPUs during training

    for epoch in range(num_epochs):
        # NEW: Set sampler to ensure each epoch has a different shuffle order
        train_loader.sampler.set_epoch(epoch)

        model.train()
        for features, labels in train_loader:

            features, labels = features.to(device), labels.to(device)  # New: use device for cpu
            logits = model(features)
            loss = F.cross_entropy(logits, labels)  # Loss function

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # LOGGING
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    model.eval()

    try:
        train_acc = compute_accuracy(model, train_loader, device=device)
        print(f"[GPU{rank}] Training accuracy", train_acc)
        test_acc = compute_accuracy(model, test_loader, device=device)
        print(f"[GPU{rank}] Test accuracy", test_acc)

    ####################################################
    # NEW:
    except ZeroDivisionError as e:
        raise ZeroDivisionError(
            f"{e}\n\nThis script is designed for 2 GPUs. You can run it as:\n"
            "torchrun --nproc_per_node=2 DDP-script-torchrun.py\n"
            f"Or, to run it on {torch.cuda.device_count()} GPUs, uncomment the code on lines 103 to 107."
        )
    ####################################################

    destroy_process_group()  
    #  NEW: cleanly exit distributed mode
    #  properly shut down the distributed training processes and release associated resources.


def compute_accuracy(model, dataloader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()


'''
When running the script using torchrun, it automatically launches one process per GPU and assigns each process a unique rank, 
along with other distributed training metadata (like world size and local rank), which are passed into the script via environment variables. 
In the __main__ block, we read these variables using os.environ and pass them to the main() function.
'''
if __name__ == "__main__":
    # NEW: Use environment variables set by torchrun if available, otherwise default to single-process.
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1

    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = 0

    '''
    ✅ gloo backend works — supports DDP across CPU processes
    ❌ nccl backend — NVIDIA only, won't work
    ❌ MPS (Metal) — Apple's GPU backend does not support distributed training at all
    '''
    if rank == 0:
        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("MPS available:", hasattr(torch, "mps") and torch.mps.is_available())
        print("CUDA GPUs available:", torch.cuda.device_count())
        print("platform.system():", platform.system())

    torch.manual_seed(123)
    num_epochs = 3
    main(rank, world_size, num_epochs)


    '''
(.venv) neoli@Neo-Mars pytorch % torchrun --nproc_per_node=2 ./DDP-script-torchrun.py
W0309 22:42:40.302000 53351 torch/distributed/elastic/multiprocessing/redirects.py:29] NOTE: Redirects are currently not supported in Windows or MacOs.
W0309 22:42:40.311000 53351 torch/distributed/run.py:852] 
W0309 22:42:40.311000 53351 torch/distributed/run.py:852] *****************************************
W0309 22:42:40.311000 53351 torch/distributed/run.py:852] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0309 22:42:40.311000 53351 torch/distributed/run.py:852] *****************************************
PyTorch version: 2.10.0
CUDA available: False
MPS available: True
CUDA GPUs available: 0
platform.system(): Darwin
[GPU0] Epoch: 001/003 | Batchsize 002 | Train/Val Loss: 0.66
[GPU0] Epoch: 002/003 | Batchsize 002 | Train/Val Loss: 0.24
[GPU0] Epoch: 003/003 | Batchsize 002 | Train/Val Loss: 0.32
[GPU0] Training accuracy 1.0
[GPU0] Test accuracy 1.0
[GPU1] Epoch: 001/003 | Batchsize 002 | Train/Val Loss: 0.67
[GPU1] Epoch: 002/003 | Batchsize 002 | Train/Val Loss: 0.47
[GPU1] Epoch: 003/003 | Batchsize 002 | Train/Val Loss: 0.18
[GPU1] Training accuracy 1.0
[GPU1] Test accuracy 1.0
    '''