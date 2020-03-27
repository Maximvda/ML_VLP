

from experiments.experiment1 import experiment1
from experiments.experiment2 import experiment2
from experiments.experiment3 import experiment3



def experiment(args):
    if args.experiment == 1:
        experiment1(args)
    elif args.experiment == 2:
        experiment2(args)
    elif args.experiment == 3:
        experiment3(args)
