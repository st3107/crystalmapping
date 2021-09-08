import typing
from collections import ChainMap

import event_model as em
import numpy as np
import pandas as pd
from bluesky.callbacks.stream import LiveDispatcher
from databroker import Broker, Header
from trackpy import link, locate

from crystalmapping._vend import gen_stream


def gen_processed_images(images: typing.Iterable[typing.Union[list, np.ndarray]],
                         subtrahend: typing.Union[list, np.ndarray]) -> typing.Generator[np.ndarray, None, None]:
    """Generate processed image from a series of images.

    The process procedure is (1) turn ot numpy array, (2) average the frames to a two dimensional image, (3)
    subtract the image and fill zero in negative pixels.

    Parameters
    ----------
    images :
        A iterable of images. The dimensions of each image is no less than 2.
    subtrahend :
        The subtrahend image. The dimensions of each image is no less than 2.

    Yields
    ------
    processed_image :
        A two dimensional image.
    """
    subtrahend = np.asarray(subtrahend)
    subtrahend = get_mean_frame(subtrahend)
    for image in images:
        image = np.asarray(image)
        image = get_mean_frame(image)
        image = subtract_image(image, subtrahend)
        yield image


def get_mean_frame(frames: np.ndarray) -> np.ndarray:
    """Average the frames to a two dimensional image."""
    n = np.ndim(frames)
    if n < 2:
        raise ValueError("The dimension of {} < 2.".format(n))
    elif n == 2:
        mean_frame = np.copy(frames)
    elif n == 3:
        mean_frame = np.mean(frames, axis=0)
    else:
        mean_frame = np.mean(frames, axis=tuple((i for i in range(n - 2))))
    return mean_frame


def subtract_image(minuend: np.ndarray, subtrahend: np.ndarray) -> np.ndarray:
    """Subtract the image and fill zero in negative pixels."""
    ans = np.zeros_like(minuend)
    np.subtract(minuend, subtrahend, out=ans, where=ans > 0)
    return ans


def get_dataframe(run: Header) -> pd.DataFrame:
    """Get the dataframe from the stream. Drop the time column."""
    return run.table().drop(columns=["time"])


class ImageProcessor(LiveDispatcher):
    """A callback to average frames of images, subtract it by another image, and emit the document."""

    def __init__(self, data_key: str, subtrahend: np.ndarray):
        """Initiate the instance.

        Parameters
        ----------
        data_key :
            The key of the data to use.
        subtrahend :
            The 2d image as a subtrahend.
        """
        super(ImageProcessor, self).__init__()
        self.data_key = data_key
        self.subtrahend = np.asarray(subtrahend)

    def start(self, doc, _md=None):
        if _md is None:
            _md = {}
        _md = ChainMap({"analysis_stage": ImageProcessor.__name__}, _md)
        super(ImageProcessor, self).start(doc, _md=_md)

    def event_page(self, doc):
        for event_doc in em.unpack_event_page(doc):
            self.event(event_doc)

    def event(self, doc, **kwargs):
        frames = np.asarray(doc["data"][self.data_key])
        minuend = get_mean_frame(frames)
        result = subtract_image(minuend, self.subtrahend)
        new_data = {k: v for k, v in doc["data"].items() if k != self.data_key}
        new_data[self.data_key] = result.tolist()
        self.process_event({'data': new_data, 'descriptor': doc["descriptor"]})


class PeakTracker(LiveDispatcher):
    """Track the peaks on a series of images and summarize their position and intensity in a dataframe."""

    def __init__(self, data_key: str, diameter: typing.Union[int, tuple], **kwargs):
        """Initiate the instance.

        Parameters
        ----------
        data_key :
            The key of the data to use.
        diameter :
            The pixel size of the peak.
        kwargs :
            The other kwargs for the `trackpy.locate`.
        """
        kwargs["diameter"] = diameter
        super(PeakTracker, self).__init__()
        self.data_key = data_key
        self.config = kwargs

    def start(self, doc, _md=None):
        _md = {"analysis_stage": PeakTracker.__name__}
        super(PeakTracker, self).start(doc, _md=_md)

    def event_page(self, doc):
        for event_doc in em.unpack_event_page(doc):
            self.event(event_doc)

    def event(self, doc, **kwargs):
        image = doc["data"][self.data_key]
        df = locate(image, **self.config)
        df = df.assign(frame=doc["seq_num"])
        for data in df.to_dict("records"):
            self.process_event({"data": data, "descriptor": doc["descriptor"]})


class TrackLinker(LiveDispatcher):
    """Track the peaks in frame and link them in trajectories.

    When a stop is received, the data will be pulled from the databroker and processed. Then, the dataframe will
    be emitted row by row.
    """

    def __init__(self, *, db: Broker = None, search_range: typing.Union[float, tuple], **kwargs):
        """Create the instance.

        Parameters
        ----------
        db :
            The databroker. If None, this callback does nothing.
        search_range :
            The search_range in `trackpy.link`.
        kwargs :
            Other kwargs in `trackpy.link`.
        """
        super(TrackLinker, self).__init__()
        kwargs["search_range"] = search_range
        self.config = kwargs
        self.db = db

    def start(self, doc, _md=None):
        _md = {"analysis_stage": TrackLinker.__name__}
        super(TrackLinker, self).start(doc, _md=_md)

    def event_page(self, doc):
        return

    def event(self, doc, **kwargs):
        return

    def stop(self, doc, _md=None):
        features = get_dataframe(self.db[doc["run_start"]])
        df = link(features, **self.config)
        descriptor = next(iter(self.raw_descriptors.keys()))
        for data in df.to_dict("records"):
            self.process_event({"data": data, "descriptor": descriptor})
        super(TrackLinker, self).stop(doc, _md=None)


class DataFrameDumper(object):
    """Dump the dataframe to the database using databroker."""

    def __init__(self, db: Broker):
        """Create an instance"""
        super(DataFrameDumper, self).__init__()
        self._db = db

    def dump_df(self, df: pd.DataFrame, metadata: dict = None):
        """Dump the data frame into the database with the metadata."""
        if not metadata:
            metadata = {}
        data = df.to_dict("records")
        for name, doc in gen_stream(data, metadata):
            self._db.insert(name, doc)
