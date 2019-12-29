import traceback

from utils.config import parse_args
from models.model import model_obj
from eval.eval import eval_obj

def main(args):
    if args.is_train:
        model = model_obj(args)
        model.train(args)
        return model.get_distance()
    else:
        #Best performing model is loaded and evaluated on the test set
        evalObj = eval_obj(args)
        #evalObj.demo([[500,750],[1000,1750]], [[100,250],[4000,2750]])
        return evalObj.demo()

if __name__ == '__main__':
    print("Python code started")
    try:
        args = parse_args()
        print("Arguments parsed")
        if args.experiment == None:
            main(args)
        else:
            from utils.experiments import experiment
            experiment(args)

    except Exception as e:
        print(e)
        traceback.print_exc()
