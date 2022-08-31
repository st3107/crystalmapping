from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback

import crystalmapping.plans as plans
import crystalmapping.sim as sim


def create_RE():
    bec = BestEffortCallback()
    bec.disable_plots()
    bec.disable_baseline()
    RE = RunEngine()
    RE.subscribe(bec, "all")
    return RE


def test_fly_scan_2d():
    RE = create_RE()
    shutter = sim.SynSignal(name="shutter")
    area_det = sim.FakeAreaDetector(name="dexela")
    area_det.cam.configure({"acquire_time": 1.})
    motor_y = sim.DelayedSynAxis(name="motor_y")
    motor_y.configure({"velocity": 1.})
    motor_x = sim.DelayedSynAxis(name="motor_x")
    motor_x.configure({"velocity": 1.})

    plan = plans.fly_scan_nd(
        [area_det], motor_y, 0., 1., 2, motor_x, 0., 2., 3,
        time_per_point=1., time_per_frame=1., move_velocity=1., shutter=shutter, shutter_close=1, shutter_open=0.,
        shutter_wait_close=0., shutter_wait_open=0.
    )
    RE(plan)


def test_fly_scan_3d():
    RE = create_RE()
    shutter = sim.SynSignal(name="shutter")
    area_det = sim.FakeAreaDetector(name="dexela")
    area_det.cam.configure({"acquire_time": 1.})
    motor_y = sim.DelayedSynAxis(name="motor_y")
    motor_y.configure({"velocity": 1.})
    motor_z = sim.DelayedSynAxis(name="motor_z")
    motor_z.configure({"velocity": 1.})
    motor_x = sim.DelayedSynAxis(name="motor_x")
    motor_x.configure({"velocity": 1.})

    plan = plans.fly_scan_nd(
        [area_det], motor_z, 2., 3., 2, motor_y, 1., 2., 2, motor_x, 0., 1., 2,
        time_per_point=1., time_per_frame=1., move_velocity=1., shutter=shutter, shutter_close=1, shutter_open=0.,
        shutter_wait_close=0., shutter_wait_open=0.
    )
    RE(plan)


def test_grid_scan_2d():
    RE = create_RE()
    shutter = sim.SynSignal(name="shutter")
    area_det = sim.FakeAreaDetector(name="dexela")
    area_det.cam.configure({"acquire_time": 1.})
    motor_y = sim.DelayedSynAxis(name="motor_y")
    motor_y.configure({"velocity": 1.})
    motor_x = sim.DelayedSynAxis(name="motor_x")
    motor_x.configure({"velocity": 1.})
    plan = plans.grid_scan_nd(
        [area_det], motor_y, 0., 1., 2, motor_x, 0., 2., 3,
        time_per_point=1., time_per_frame=1., shutter=shutter, shutter_close=1., shutter_open=0.,
        shutter_wait_close=0., shutter_wait_open=0.
    )
    RE(plan)


def test_loop_until():
    RE = create_RE()
    motor_x = sim.DelayedSynAxis(name="motor_x")
    motor_x.configure({"velocity": 10.})
    plan = plans.loop_forever(motor_x, 0, 1, 0.2)
    RE(plan)
