import dask
import pandas as pd
from databroker import Broker

DB = Broker.named("xpd")
UID = pd.read_csv("data/uid.csv")
dask.config.set(scheduler='threads')
DB2 = Broker.named("analysis")
