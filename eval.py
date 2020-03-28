# from database import Record, Pair, Logger
# import pathlib

from dotenv import load_dotenv
load_dotenv('env.list')
import os
import subprocess
from datetime import datetime
import pandas as pd

# pathlib.Path(os.environ['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
application = os.environ['TEST_NAME'] + '_' + os.environ['DATA_DIR'] + '_' + str(int(datetime.now().timestamp()))
# logger = Logger(application)

def run_model(phrase):
    os.chdir('./neural')
    command = "docker exec -it sotaws sh -c 'luajit tokenizer-neuron.lua -model {0} \"{1}\"'".format(os.environ['MODEL_DIR'], phrase).split(' ')
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    print(result.returncode, result.stdout, result.stderr)
    os.chdir('..')

def iterate_dataframe(df):
    for i, row in df.iterrows():
        token = df.at[i, 'joined_ngram']
        try:
            ori_seg = df.at[i, 'original_n_gram']
        except:
            ori_seg = df.at[i, 'original_ngram']
        record = {
            "token": token,
            "ori_seg": ori_seg, 
            "des_seg": run_model(token)
        }
        # logger.add_pair(record)


if __name__ == '__main__':
    filename = os.path.join(os.environ['DATA_DIR'], os.environ['FILENAME'])
    df = pd.read_csv(filename)
    iterate_dataframe(df)