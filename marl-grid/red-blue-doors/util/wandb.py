import wandb
from torch.multiprocessing import Process, Queue


class WandbLoggingProcess(Process):
    def __init__(self, queue: Queue, *args, **kwargs):
        super(WandbLoggingProcess, self).__init__()
        self.queue = queue
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        wandb.init(*self.args, **self.kwargs)
        while True:
            log_dict = self.queue.get(block=True)
            wandb.log({**log_dict, "queue_size": self.queue.qsize()})
