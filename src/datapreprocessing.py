import pandas as pd
import zipfile

class DataPreprocessor:
    def __init__(self):
        self.dataset = pd.DataFrame()
        # self.train = pd.DataFrame()
        # self.validation = pd.DataFrame()
        # self.test = pd.DataFrame()
    
    def load_data(self, dataset_name: str):
            '''
            Loads the specified dataset into the object.

            Parameters:
            dataset_name (str): The name of the dataset to load. Should be either "retailrocket" or "tmall".

            Raises:
            ValueError: If an invalid dataset name is provided.

            Notes:
            - The Retailrocket original dataset can be found at: https://www.kaggle.com/retailrocket/ecommerce-dataset
            - The Tmall original dataset can be found at: https://tianchi.aliyun.com/dataset/43
            '''
            if dataset_name == "retailrocket":
                    self.dataset = pd.read_csv(r"data\retailrocket\unprocessed\events.zip")
            elif dataset_name == "tmall":
                    self.dataset = pd.read_csv(r"data\tmall\unprocessed\dataset15.zip", sep='\t')
            else:
                    raise ValueError("Invalid dataset name. Please choose 'retailrocket' or 'tmall'.") 
    