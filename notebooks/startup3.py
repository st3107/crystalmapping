import trackpy as tp
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from databroker import Broker
from pymongo import MongoClient
from csvdb.client import DataFrameClient, DataSetClient


mongo_client = MongoClient()
db_raw = Broker.named("xpd")
db_ana = Broker.named("analysis")
df_uid = pd.read_csv("./data/uid.csv")
db_csv = DataFrameClient(mongo_client.csv_db, "/Volumes/STAO_EXT/csv_db_data_external")
db_cdf = DataSetClient(mongo_client.cdf_db, "/Volumes/STAO_EXT/cdf_db_data_external")
