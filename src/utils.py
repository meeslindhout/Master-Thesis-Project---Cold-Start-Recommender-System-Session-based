'''
Code has been sourced from the following repository:
Malte. (2024). Rn5l/session-rec [Python]. https://github.com/rn5l/session-rec (Original work published 2019)
(https://github.com/rn5l/session-rec/blob/master/algorithms/knn/sknn.py)
'''
import pandas as pd
import os
import time
import dill
from pathlib import Path

def ensure_dir(file_path):
    '''
    Create all directories in the file_path if non-existent.
        --------
        file_path : string
            Path to the a file
    '''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def save_model(key, algorithm, conf):
    '''
    Save the model object for reuse with FileModel
        --------
        algorithm : object
            Dictionary of all results res[algorithm_key][metric_key]
        conf : object
            Configuration dictionary, has to include results.pickel_models
    '''

    file_name = conf['results']['folder'] + '/' + conf['key'] + '_' + conf['data']['name'] + '_' + key + '.pkl'
    file_name = Path(file_name)
    ensure_dir(file_name)
    file = open(file_name, 'wb')

    # pickle.dump(algorithm, file)
    dill.dump(algorithm, file)

    file.close()

def write_results_csv(results, conf, iteration=None, extra=None):
    '''
    Write the result array to a csv file, if a result folder is defined in the configuration
        --------
        results : dict
            Dictionary of all results res[algorithm_key][metric_key]
        iteration; int
            Optional for the window mode
        extra: string
            Optional string to add to the file name
    '''

    if 'results' in conf and 'folder' in conf['results']:

        export_csv = conf['results']['folder'] + 'test_' + conf['type'] + '_' + conf['key'] + '_' + conf['data']['name']
        # if extra is not None:
        #     export_csv += '.' + str(extra)
        if iteration is not None:
            export_csv += '.' + str(iteration)
        export_csv += '.csv'

        ensure_dir(export_csv)

        file = open(export_csv, 'w+')
        file.write('Metrics;')

        for k, l in results.items():
            for e in l:
                file.write(e[0])
                file.write(';')
            break

        file.write('\n')

        for k, l in results.items():
            file.write(k)
            file.write(';')
            for e in l:
                file.write(str(e[1]))
                file.write(';')
                if len(e) > 2:
                    if type(e[2]) == pd.DataFrame:
                        name = export_csv.replace('.csv', '-') + e[0].replace(':', '').replace(' ', '') + '.csv'
                        e[2].to_csv(name, sep=";", index=False)
            file.write('\n')


def eval_algorithm(train, test, key, algorithm, eval, metrics, results, conf, slice=None, iteration=None, out=True):
    '''
    Evaluate one single algorithm
        --------
        train : Dataframe
            Training data
        test: Dataframe
            Test set
        key: string
            The automatically created key string for the algorithm
        algorithm: algorithm object
            Just the algorithm object, e.g., ContextKNN
        eval: module
            The module for evaluation, e.g., evaluation.evaluation_last
        metrics: list of Metric
            Optional string to add to the file name
        results: dict
            Result dictionary
        conf: dict
            Configuration dictionary
        slice: int
            Optional index for the window slice
    '''
    ts = time.time()
    print('fit ', key)
    # send_message( 'training algorithm ' + key )

    if hasattr(algorithm, 'init'):
        algorithm.init(train, test, slice=slice)

    for m in metrics:
        if hasattr(m, 'start'):
            m.start(algorithm)

    algorithm.fit(train, test)
    print(key, ' time: ', (time.time() - ts))

    if 'results' in conf and 'pickle_models' in conf['results']:
        try:
            save_model(key, algorithm, conf)
        except Exception:
            print('could not save model for ' + key)

    for m in metrics:
        if hasattr(m, 'start'):
            m.stop(algorithm)

    results[key] = eval.evaluate_sessions(algorithm, metrics, test, train)
    if out:
        write_results_csv({key: results[key]}, conf, extra=key, iteration=iteration)

    # send_message( 'algorithm ' + key + ' finished ' + ( 'for slice ' + str(slice) if slice is not None else '' ) )

    algorithm.clear()