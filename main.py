import traceback

from utils.config import parse_args
from dataset.setup_database import setup_database
from models.CNN import CNN

def main(args):
    data_loader = setup_database(args)

    model = CNN(args,data_loader)
    if args.is_train:
        model.train(args)
    return

def evaluate(args):
    return

if __name__ == '__main__':
    print("Python code started")
    try:
        args = parse_args()
        print("Arguments parsed")
        if args.is_train:
            main(args)
        else:
            evaluate(args)
    except Exception as e:
        print(e)
        traceback.print_exc()
