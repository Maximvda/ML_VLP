import traceback

from utils.config import parse_args
from dataset.setup_database import setup_database
from models.CNN import CNN
from eval.eval import eval

def main(args):
    data_loader = setup_database(args)

    model = CNN(args,data_loader)
    if args.is_train:
        model.train(args)
    else:
        eval_obj = eval(args, model.getModel())
        eval_obj.demo()
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
