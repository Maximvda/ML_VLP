import torch
import torch.multiprocessing as _mp
from trainers.trainer import Trainer

mp = _mp.get_context('spawn')

#Worker class that trains population task and calculates performance
class Worker(mp.Process):
    def __init__(self, args, worker_id, tasks, par_list):
        super().__init__()
        self.tasks = tasks
        if type(args.gpu_number) == list:
            args.device = torch.device('cuda', worker_id % len(args.gpu_number)+args.gpu_number[0])
        else:
            args.device = torch.device('cuda', args.gpu_number)
        self.trainer = Trainer(args,worker_id=worker_id)
        self.par_list = par_list

    def run(self):
        # Train model
        task = self.tasks.get()
        par_dict = self.par_list[task['id']]
        for key in par_dict:
            setattr(self, key, par_dict[key])

        self.trainer.set_id(task['id'])
        #Train the model till no performance improvement
        self.trainer.train(experiment=True)
