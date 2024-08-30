import torch
import json
import numpy as np
import matplotlib.pyplot as plt
def getopt(opt, key, default_value=None):
        if default_value is None and (opt is None or key not in opt):
            raise ValueError(f'Error: required key {key} was not provided in an opt.')
        return opt[key] if opt and key in opt else default_value

def write_json(file, path):
    with open(path, 'w') as f:
        f.write('[')
        for i, item in enumerate(file):
            json.dump(item,f)
            if i != len(file)-1:
                f.write(',\n')
            else:
                f.write(']\n')

def display_logs(file, model_name, save=False):

    losses = [o['loss_results'] for o in file]
    step = file[0]['best_iter'] + 1
    steps = np.arange(step, len(file) * step + 1, step)
    meteor = [o['ap_results']['meteor'] for o in file]
    fig, ax = plt.subplots(2, 1, sharex='col')
    ax[0].plot(steps, losses, 'bo-')
    ax[0].set_ylabel('loss')
    ax[0].set_title('Loss and METEOR score during training, on evaluation dataset')
    ax[1].plot(steps, meteor, 'go-')
    ax[1].set_ylabel('METEOR')
    fig.text(.5, .04, 'iter')
    if save:
        plt.savefig('AlexCap/graphs/'+model_name+'.png')
    plt.show()