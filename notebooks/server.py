import numpy as np
from databroker import Broker
from bluesky.callbacks.best_effort import BestEffortCallback
from crystalmapping.callbacks import ImageProcessor, PeakTrackor, TrackLinker
from bluesky.callbacks.zmq import RemoteDispatcher

# load the image to use
image = np.load("./data/PARAMID-2_background_1.npy", allow_pickle=True)

# create a remote dispatcher to receive the dispatch the data
rd = RemoteDispatcher(address=("localhost", 5568), prefix=b'raw')

# create the callback
ip = ImageProcessor("dexela_image", subtrahend=image)
pt = PeakTracker("dexela_image", diameter=15, percentile=80, separation=100)
tl = TrackLinker(db=DB2, search_range=50)
bec = BestEffortCallback()
bec.disable_plots()
tl.subscribe(DB2.insert)
pt.subscribe(tl)
pt.subscribe(DB2.insert)
ip.subscribe(pt)
ip.subscribe(bec)

# subscribe the callback to the dispatcher
rd.subscribe(ip)

# start the server
if __name__ == "__main__":
    print("Start the server ...")
    rd.start()
