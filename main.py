import traceback

from utils.config import parse_args
from utils.experiments import runExperiment
from models.CNN import CNN
from eval.eval import eval_obj

def main(args):
    if args.is_train:
        model = CNN(args)
        model.train(args)
        return model.get_distance()
    else:
        #Best performing model is loaded and evaluated on the test set
        evalObj = eval_obj(args)
        return evalObj.demo()

if __name__ == '__main__':
    print("Python code started")
    try:
        args = parse_args()
        print("Arguments parsed")
        if args.experiment == None:
            main(args)
        else:
            runExperiment(args)

    except Exception as e:
        print(e)
        traceback.print_exc()
