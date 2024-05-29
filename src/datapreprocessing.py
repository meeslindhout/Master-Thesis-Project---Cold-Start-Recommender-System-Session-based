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
            - Datasets can be downloaded from Google Drive and should be stored as zips in the "unprocessed" folder.
                The Google Drive link can be found in the documentation.
            - The Retailrocket original dataset can be found at: https://www.kaggle.com/retailrocket/ecommerce-dataset
            - The Tmall original dataset can be found at: https://tianchi.aliyun.com/dataset/43
            - The Google Drive is hosted by Malte. (2024). Rn5l/session-rec [Python]. https://github.com/rn5l/session-rec (Original work published 2019)
                and created for convenience.
            '''
            if dataset_name == "retailrocket":
                    self.dataset = pd.read_csv(r"data\retailrocket\unprocessed\events.csv")
            elif dataset_name == "tmall":
                    self.dataset = pd.read_csv(r"data\tmall\unprocessed\dataset15.csv", nrows=100, sep='\t')
            else:
                    raise ValueError("Invalid dataset name. Please choose 'retailrocket' or 'tmall'.") 
    