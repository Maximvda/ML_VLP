import traceback

from utils.config import parse_args
from dataset.setup_database import setup_database
from models.CNN import CNN
from eval.eval import eval

def main(args):
    if args.is_train:
        data_loader = setup_database(args)
        model = CNN(args,data_loader)
        model.train(args)
    else:
        #from utils.initModel import initModel
        #val_data_loader = setup_database(args, 'val')
        #md = initModel(val_data_loader, args.nf).to(args.device)
        #import os
        #pth = os.path.join(args.result_root,'evalStats.pth')
        #import torch
        #chk = torch.load(pth, map_location=args.device)
        #md.load_state_dict(chk['best_model'])
        #distance = []
        #from utils.utils import calcDistance
        #for i, data in enumerate(val_data_loader):
        #    with torch.no_grad():
        #        #forward batch of test data through the network
        #        input = data[0].type(torch.FloatTensor).to(args.device)
        #        output = data[1].to(args.device)
        #        prediction = md(input)[:,:,0,0]
        #        distance.append(calcDistance(prediction, output))
                #visualise(output, prediction, pause=0.1)

        #The average distance over the entire test set is calculated
        #dist = sum(distance)/len(distance)
        #The distance is denormalised to cm's
        #dist = dist*300
        #print("In epoch: {}\tBest distance: {}\nDistance on test set: {}cm".format(1, 10, dist))
        eval_obj = eval(args)
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
