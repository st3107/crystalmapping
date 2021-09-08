import numpy as np
from ophyd import Device, Component, Kind
from ophyd.sim import DirectImage, SynSignal, SynAxis


class FakeCam(DirectImage):
    acquire_time = Component(SynSignal, kind=Kind.config)


class FakeAreaDetector(Device):
    cam = Component(FakeCam, func=lambda: np.ones(5, 5), kind=Kind.normal)
    images_per_set = Component(SynSignal, kind=Kind.config)

    def trigger(self):
        self.cam.img.exposure_time = self.cam.acquire_time.value * self.images_per_set.value
        return super(FakeAreaDetector, self).trigger()


class DelayedSynAxis(SynAxis):

    def set(self, value):
        self.delay = abs(self.position - value) / self.velocity.value
        return super(DelayedSynAxis, self).set(value)
