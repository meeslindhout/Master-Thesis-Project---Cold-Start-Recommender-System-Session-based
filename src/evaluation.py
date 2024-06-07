'''
Code has been sourced from the following repository:
Malte. (2024). Rn5l/session-rec [Python]. https://github.com/rn5l/session-rec (Original work published 2019)
(https://github.com/rn5l/session-rec/blob/5dcd583cbd8d44703a5248b9a308945f24b91390/evaluation/evaluation.py#L238)
(https://github.com/rn5l/session-rec/blob/master/evaluation/metrics/accuracy.py)
(https://github.com/rn5l/session-rec/blob/master/algorithms/hybrid/strategic.py)
'''
import numpy as np
# import time

def evaluate_sessions(pr, metrics, test_data, train_data, items=None, cut_off=20, session_key='SessionId', item_key='ItemId', time_key='Time'): 
    '''
    Evaluates the baselines wrt. recommendation accuracy measured by recall@N and MRR@N. Has no batch evaluation capabilities. Breaks up ties.

    Parameters
    --------
    pr : baseline predictor
        A trained instance of a baseline predictor.
    metrics : list
        A list of metric classes providing the proper methods
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    train_data : pandas.DataFrame
        Training data. Only required for selecting the set of item IDs of the training set.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    
    Returns
    --------
    out :  list of tuples
        (metric_name, value)
    '''
    
    # Calculate the total number of actions and sessions in the test data
    actions = len(test_data)
    sessions = len(test_data[session_key].unique())
    count = 0
    print('START evaluation of ', actions, ' actions in ', sessions, ' sessions')
    
    # Reset all metrics before evaluation
    for m in metrics:
        m.reset()
    
    # Sort the test data by session ID and timestamp
    test_data.sort_values([session_key, time_key], inplace=True)
    
    # Get the unique item IDs from the training data
    items_to_predict = train_data[item_key].unique()
    
    # Initialize variables for tracking previous item ID and session ID
    prev_iid, prev_sid = -1, -1
    pos = 0
    
    # Iterate through each action in the test data
    for i in range(len(test_data)):
        
        # Print progress every 1000 actions
        if count % 1000 == 0:
            print( '    eval process: ', count, ' of ', actions, ' actions: ', ( count / actions * 100.0 ), ' %')
        
        # Get the session ID, item ID, and timestamp for the current action
        sid = test_data[session_key].values[i]
        iid = test_data[item_key].values[i]
        ts = test_data[time_key].values[i]
        
        # Check if the session ID has changed
        if prev_sid != sid:
            prev_sid = sid
            pos = 0
        else:
            # Check if specific items are provided for prediction
            if items is not None:
                if np.in1d(iid, items):
                    items_to_predict = items
                else:
                    items_to_predict = np.hstack(([iid], items))  
                    
            # Call the start_predict method for each metric
            for m in metrics:
                if hasattr(m, 'start_predict'):
                    m.start_predict(pr)
            
            # Predict the next item using the baseline predictor
            preds = pr.predict_next(sid, prev_iid, items_to_predict, #timestamp=ts (is not required)
                                    )
            
            # Call the stop_predict method for each metric
            for m in metrics:
                if hasattr(m, 'stop_predict'):
                    m.stop_predict(pr)
            
            # Replace NaN values with 0 and sort the predictions in descending order
            preds[np.isnan(preds)] = 0
            preds.sort_values(ascending=False, inplace=True)
            
            # Call the add method for each metric
            for m in metrics:
                if hasattr(m, 'add'):
                    m.add(preds, iid, for_item=prev_iid, session=sid, position=pos)
            
            pos += 1
            
        prev_iid = iid
        
        count += 1

    print('END evaluation')

    # Get the results for each metric
    res = []
    for m in metrics:
        res.append(m.result())

    return res

class MRR: 
    '''
    MRR( length=20 )

    Used to iteratively calculate the average mean reciprocal rank for a result list with the defined length. 

    Parameters
    -----------
    length : int
        MRR@length
    '''
    def __init__(self, length=20):
        self.length = length;
    
    def init(self, train):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        return
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.test=0;
        self.pos=0
        
        self.test_popbin = {}
        self.pos_popbin = {}
        
        self.test_position = {}
        self.pos_position = {}
    
    def skip(self, for_item = 0, session = -1 ):
        pass
        
    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None ):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        res = result[:self.length]
        
        self.test += 1
        
        if pop_bin is not None:
            if pop_bin not in self.test_popbin:
                self.test_popbin[pop_bin] = 0
                self.pos_popbin[pop_bin] = 0
            self.test_popbin[pop_bin] += 1
        
        if position is not None:
            if position not in self.test_position:
                self.test_position[position] = 0
                self.pos_position[position] = 0
            self.test_position[position] += 1
        
        if next_item in res.index:
            rank = res.index.get_loc( next_item )+1
            self.pos += ( 1.0/rank )
            
            if pop_bin is not None:
                self.pos_popbin[pop_bin] += ( 1.0/rank )
            
            if position is not None:
                self.pos_position[position] += ( 1.0/rank )
                   
        
        
    def add_batch(self, result, next_item):
        '''
        Update the metric with a result set and the correct next item.
        
        Parameters
        --------
        result: pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.
        next_item: Array of correct next items
        '''
        i=0
        for part, series in result.iteritems(): 
            result.sort_values( part, ascending=False, inplace=True )
            self.add( series, next_item[i] )
            i += 1
        
    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("MRR@" + str(self.length) + ": "), (self.pos/self.test), self.result_pop_bin(), self.result_position()
    
    def result_pop_bin(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        csv = ''
        csv += 'Bin: ;'
        for key in self.test_popbin:
            csv += str(key) + ';'
        csv += '\nPrecision@' + str(self.length) + ': ;'
        for key in self.test_popbin:
            csv += str( self.pos_popbin[key] / self.test_popbin[key] ) + ';'
            
        return csv
    
    def result_position(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        csv = ''
        csv += 'Pos: ;'
        for key in self.test_position:
            csv += str(key) + ';'
        csv += '\nPrecision@' + str(self.length) + ': ;'
        for key in self.test_position:
            csv += str( self.pos_position[key] / self.test_position[key] ) + ';'
            
        return csv
    
class HitRate: 
    '''
    MRR( length=20 )

    Used to iteratively calculate the average hit rate for a result list with the defined length. 

    Parameters
    -----------
    length : int
        HitRate@length
    '''
    
    def __init__(self, length=20):
        self.length = length;
    
    def init(self, train):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        return
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.test=0;
        self.hit=0
        
        self.test_popbin = {}
        self.hit_popbin = {}
        
        self.test_position = {}
        self.hit_position = {}
        
    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        
        self.test += 1
         
        if pop_bin is not None:
            if pop_bin not in self.test_popbin:
                self.test_popbin[pop_bin] = 0
                self.hit_popbin[pop_bin] = 0
            self.test_popbin[pop_bin] += 1
        
        if position is not None:
            if position not in self.test_position:
                self.test_position[position] = 0
                self.hit_position[position] = 0
            self.test_position[position] += 1
                
        if next_item in result[:self.length].index:
            self.hit += 1
            
            if pop_bin is not None:
                self.hit_popbin[pop_bin] += 1
            
            if position is not None:
                self.hit_position[position] += 1
            
        
        
    def add_batch(self, result, next_item):
        '''
        Update the metric with a result set and the correct next item.
        
        Parameters
        --------
        result: pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.
        next_item: Array of correct next items
        '''
        i=0
        for part, series in result.iteritems(): 
            result.sort_values( part, ascending=False, inplace=True )
            self.add( series, next_item[i] )
            i += 1
        
    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("HitRate@" + str(self.length) + ": "), (self.hit/self.test), self.result_pop_bin(), self.result_position()

    
    def result_pop_bin(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        csv = ''
        csv += 'Bin: ;'
        for key in self.test_popbin:
            csv += str(key) + ';'
        csv += '\nHitRate@' + str(self.length) + ': ;'
        for key in self.test_popbin:
            csv += str( self.hit_popbin[key] / self.test_popbin[key] ) + ';'
            
        return csv
    
    def result_position(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        csv = ''
        csv += 'Pos: ;'
        for key in self.test_position:
            csv += str(key) + ';'
        csv += '\nHitRate@' + str(self.length) + ': ;'
        for key in self.test_position:
            csv += str( self.hit_position[key] / self.test_position[key] ) + ';'
            
        return csv

class StrategicHybrid:
    '''
    StrategicHybrid(algorithms, weights)

    Use different algorithms depending on the length of the current session.

    Parameters
    --------
    algorithms : list
        List of algorithms to combine with a switching strategy.
    thresholds : float
        Proper list of session length thresholds.
        For [5,10] the first algorithm is applied until the session exceeds a length of 5 actions, the second up to a length of 10, and the third for the rest.
    fit: bool
        Should the fit call be passed through to the algorithms or are they already trained?

    '''

    def __init__(self, algorithms, thresholds, fit=True, clearFlag=True):
        self.algorithms = algorithms
        self.thresholds = thresholds
        self.run_fit = fit
        self.clearFlag = clearFlag
    
    def init(self, train, test=None, slice=None):
        for a in self.algorithms: 
            if hasattr(a, 'init'):
                a.init( train, test, slice )
    
    def fit(self, data):
        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).

        '''
        if self.run_fit:
            for a in self.algorithms:
                a.fit(data)

        self.session = -1
        self.session_items = []

    def predict_next(self, session_id, input_item_id, predict_for_item_ids, skip=False, timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.

        '''

        if (self.session != session_id):  # new session
            self.session = session_id
            self.session_items = list()

        self.session_items.append(input_item_id)

        predictions = []
        for a in self.algorithms:
            predictions.append(a.predict_next(session_id, input_item_id, predict_for_item_ids, skip))

        for i, prediction in reversed(list(enumerate(predictions))):
            if len(self.thresholds) - 1 < i or len(self.session_items) <= self.thresholds[i]:
                final = prediction

        return final


    def clear(self):
        if(self.clearFlag):
            for a in self.algorithms:
                a.clear()