import torch.multiprocessing as mp
import wandb


class WandbLoggingProcess(mp.Process):
    def __init__(self, master, save_dir_fmt, log_queue, *args, **kwargs):
        super().__init__()
        self.master = master
        self.save_dir_fmt = save_dir_fmt
        self.log_queue = log_queue
        self.args = args
        self.kwargs = kwargs

    def run(self):
        wandb.init(*self.args, **self.kwargs)
        # wandb.run.log_code(
        #     osp.abspath(osp.join(osp.abspath(__file__), "../..")),
        #     include_fn=lambda path: path.endswith(".py"),
        # )
        while True:
            log_dict = self.log_queue.get(block=True)
            if isinstance(log_dict, dict):
                wandb.log({**log_dict, "queue_size": self.log_queue.qsize()})
            else:
                break
        # base_path = "/".join(self.save_dir_fmt.split("/")[:-1])
        # wandb.save(
        #     glob_str=f"{base_path}/*/latest*.*", base_path=base_path, policy="now"
        # )
        # wandb.save(
        #     glob_str=f"{base_path}/*/train_log.csv", base_path=base_path, policy="now"
        # )
        wandb.finish()
        print("wandb logger is done")
        return
