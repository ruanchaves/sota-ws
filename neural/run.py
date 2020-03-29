from __future__ import print_function
import multiprocessing
import subprocess 
import os
settings = {
        "model" : os.environ['MODEL_DIR'],
        "dataset": os.path.join(os.environ['DATA_DIR'], os.environ['FILENAME'])
}

def model_func(phrase):
    cmd = "luajit tokenizer-neuron.lua -model %s \"%s\"" % (settings['model'], phrase.strip())
    out = subprocess.Popen(cmd.split(' '), 
           stdout=subprocess.PIPE, 
           stderr=subprocess.STDOUT)
    stdout,stderr = out.communicate()
    if not stderr:
        print(stdout)
        return None
    else:
        return None

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=int(os.environ['PROCESSES']))
    line_array = []
    with open(settings['dataset'],'r') as f:
        for idx, line in enumerate(f):
            line_array.append(line)
    result_list = pool.map(model_func, line_array)