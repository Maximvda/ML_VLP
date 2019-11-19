import traceback

from utils.config import parse_args
from models.CNN import CNN
from eval.eval import eval_obj

def main(args):
    if args.is_train:
        model = CNN(args)
        model.train(args)
    else:
        evalObj = eval_obj(args)
        evalObj.demo()
    return

if __name__ == '__main__':
    print("Python code started")
    try:
        args = parse_args()
        print("Arguments parsed")
        main(args)

    except Exception as e:
        print(e)
        traceback.print_exc()
