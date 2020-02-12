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

    fig, axs = plt.subplots(nrows=2, ncols=3)
    #fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    for i in range(0,6):
        ax = axs.flat[i]
        img = ax.imshow(maps[i], cmap='viridis', vmin=0, vmax=45, interpolation='nearest')
        if i ==3 or i == 4 or i == 5:
            ax.set_xlabel('x-axis: (cm)')
        if i == 0 or i == 3:
            ax.set_ylabel('y-axis: (cm)')
        ax.invert_yaxis()
        ax.set_title('Conf {}'.format(i+1))

    fig.suptitle('Prediction error: (cm)',y=1.05)
    resultpath = os.path.join(pth, 'heatmap_comparison.pdf')
    plt.tight_layout()
    #fig.text(0.5, -0.05, "X-axis: (cm)", ha='center')
    #fig.text(-0.05, 0.5, "Y-axis: (cm)", va='center', rotation='vertical')
    fig.colorbar(img, ax=list(axs))
    plt.savefig(resultpath, bbox_inches='tight')
    plt.close()
