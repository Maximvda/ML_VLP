import traceback

from utils.config import parse_args
from dataset.setup_database import setup_database

def main(args):
    dataLoader = setup_database(args)
    input, output = next(iter(dataLoader))
    print(input)
    print(output)
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
