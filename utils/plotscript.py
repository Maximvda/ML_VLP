from eval.eval import eval_obj
import os
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

def plotscript(args):
    #Plots for experiment 2
    args.TX_input = 9
    args.nf = 256
    args.extra_layers = 3
    args.model_type = 'FC_expand'
    args.dynamic = True
    args.experiment=2
    pth = os.path.join(args.result_root, 'experiment_4')

    maps = []
    for i in range(1,7):
        args.TX_config = i
        args.result_root = os.path.join(pth, 'TX_config_' + str(i))
        evalObj = eval_obj(args)
        maps.append(evalObj.heatMap(args.TX_config))

    for i in range(1,7):
        plt.subplot(23+i)
        plt.imshow(maps[i-1], cmap='viridis', vmin=0, vmax=45, interpolation='nearest')
        plt.xlabel('x-axis: (cm)')
        plt.ylabel('y-axis: (cm)')
        plt.gca().invert_yaxis()

    plt.colorbar()
    plt.title(title)
    resultpath = os.path.join(pth, 'heatmap_comparison.png')
    plt.savefig(resultpath)
    plt.close()
