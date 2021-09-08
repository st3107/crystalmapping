import dask
from dask.diagnostics import ProgressBar
import trackpy as tp
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from databroker import Broker
from pymongo import MongoClient
from csvdb.client import Client

db_raw = Broker.named("xpd")
db_ana = Broker.named("analysis")
df_uid = pd.read_csv("./data/uid.csv")
db_csv = Client(MongoClient().csv_db, "/Volumes/STAO_EXT/csv_db_data_external")

del Broker, Client, MongoClient

print("Namespace:", [s for s in dir() if not s.startswith("_")])
