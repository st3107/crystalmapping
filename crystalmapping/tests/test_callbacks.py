from pprint import pformat

import databroker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp
from pkg_resources import resource_filename

from crystalmapping._vend import gen_stream
import crystalmapping.callbacks as cbs


def print_doc(name, doc):
    print(name, "\n", pformat(doc))


def test_ImageProcessor():
    """Check that ImageProcessor is correctly subtracting images."""
    data_key = "pe1_image"

    def verify(_name, _doc):
        if _name != "event":
            return
        data = _doc["data"][data_key]
        assert isinstance(data, list)
        assert np.array_equal(np.asarray(data), np.zeros((3, 3)))

    ip = cbs.ImageProcessor(data_key=data_key, subtrahend=np.ones((3, 3)))
    ip.subscribe(verify, name="event")

    frames = np.ones((2, 3, 3))
    for name, doc in gen_stream([{data_key: frames}], {}):
        ip(name, doc)


def test_gen_processed_images():
    """Test gen_processed_images."""
    images1 = (np.ones((2, 3, 3)) for _ in range(3))
    subtrahend = np.ones((3, 3))
    subtrahend[0, 0] = 2
    images2 = cbs.gen_processed_images(images1, subtrahend=subtrahend)
    for image in images2:
        assert np.array_equal(image, np.zeros((3, 3)))


def test_PeakTracker(tmpdir):
    """Check that PeakTrack and TrackLinker works without errors."""
    tp.quiet()
    # make images
    image_file = resource_filename("crystalmapping", "data/image.png")
    image = plt.imread(image_file)
    images = [image] * 3
    # check if db friendly
    db = databroker.v1.temp()
    # check features
    data_key = "pe1_image"
    pt = cbs.PeakTracker(data_key=data_key, diameter=(11, 11))
    pt.subscribe(db.insert)
    data = [{data_key: image} for image in images]
    for name, doc in gen_stream(data, {}):
        pt(name, doc)
    df = cbs.get_dataframe(db[-1])
    print(df.to_string())
    # check trajectories
    tl = cbs.TrackLinker(db=db, search_range=3)
    tl.subscribe(db.insert)
    for name, doc in db[-1].documents(fill="yes"):
        tl(name, doc)
    df = cbs.get_dataframe(db[-1])
    print(df.to_string())


def test_DataFrameDumper():
    """Test DataFrameDumper."""
    db = databroker.v1.temp()
    dfd = cbs.DataFrameDumper(db)
    data = [1, 2, 3]
    df = pd.DataFrame({"a": [1, 2, 3]})
    metadata = {"key": "a"}
    dfd.dump_df(df, metadata)
    run = db[-1]
    assert run.start["key"] == metadata["key"]
    assert list(run.data("a")) == data
