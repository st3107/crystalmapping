import trackpy as tp
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import crystalmapping.callbacks as cbs
import crystalmapping.utils as utils
from databroker import Broker
import pymongo
from csvdb.client import Client

db_raw = Broker.named("xpd")
db_ana = Broker.named("analysis")
df_uid = pd.read_csv("./data/uid.csv")
db_csv = Client(pymongo.MongoClient().csv_db, "/Volumes/STAO_EXT/csv_db_data_external")

del Broker, Client

print("Namespace:", [s for s in dir() if not s.startswith("_")])
