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
            args.device = torch.device('cuda', args.gpu_number[worker_id % len(args.gpu_number)])
        else:
            args.device = torch.device('cuda', args.gpu_number)
        self.trainer = Trainer(args,worker_id=worker_id)
        self.par_list = par_list

    def run(self):
        while True:
            if self.tasks.empty():
                break
            #Get Id of the task to perform
            task = self.tasks.get()
            #Set the hyperparameters
            par_dict = self.par_list[task['id']]
            for key in par_dict:
                self.trainer.set_attribute(key, par_dict[key])

            #Re initialise datasets if data configuration is changed
            if any([par in par_dict for par in ['TX_config', 'TX_input', 'blockage', 'output_nf', 'dataset_path']]):
                self.trainer.set_dataset()

            #Initialise model in trainer for task id
            self.trainer.set_id(task['id'])
            #Train the model till no performance improvement
            self.trainer.train(experiment=True)
