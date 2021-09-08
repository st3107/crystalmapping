import itertools
import math
import time
import typing
import uuid

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import bluesky.preprocessors as bpp
import numpy as np
import ophyd
from bluesky.utils import short_uid
from ophyd import Signal, Kind


class TomoPlanError(Exception):
    """Error in the plans."""
    pass


def _extarct_motor_pos(mtr):
    """Extract the motor position."""
    ret = yield from bps.read(mtr)
    if ret is None:
        return None
    return next(
        itertools.chain(
            (ret[k]["value"] for k in mtr.hints.get("fields", [])),
            (v["value"] for v in ret.values()),
        )
    )


def configure_area_det(detector, exposure, acq_time):
    """Configure exposure time of a detector in continuous acquisition mode."""
    if exposure < acq_time:
        raise TomoPlanError("exposure time < frame acquisition time: {} < {}".format(exposure, acq_time))
    yield from bps.mv(detector.cam.acquire_time, acq_time)
    res = yield from bps.read(detector.cam.acquire_time)
    real_acq_time = res[detector.cam.acquire_time.name]["value"] if res else 1
    if hasattr(detector, "images_per_set"):
        # compute number of frames
        num_frame = math.ceil(exposure / real_acq_time)
        yield from bps.mv(detector.images_per_set, num_frame)
    else:
        # The dexela detector does not support `images_per_set` so we just
        # use whatever the user asks for as the thing
        num_frame = 1
    computed_exposure = num_frame * real_acq_time
    return num_frame, real_acq_time, computed_exposure


def dark_plan(detector):
    """Take a dark scan in "dark" stream."""
    # Restage to ensure that dark frames goes into a separate file.
    yield from bps.unstage(detector)
    yield from bps.stage(detector)

    yield from bps.trigger_and_read([detector], name="dark")

    # Restage.
    yield from bps.unstage(detector)
    yield from bps.stage(detector)


def _get_motors_and_coords(starts, ends, nums) -> np.ndarray:
    """Get the motors and coordinates of the motors, like [[motor1_pos1, motor2_pos1, ...],
    [motor1_pos2, motor2_pos2, ...], ...]."""
    axes = [np.linspace(start, end, num) for start, end, num in zip(starts, ends, nums)]
    grids = np.meshgrid(*axes, sparse=False, indexing="ij")
    coords = np.column_stack([grid.flatten() for grid in grids])
    return coords


def fly_scan_nd(
    detectors: list,
    *args,
    move_velocity: float,
    time_per_point: float,
    time_per_frame: float,
    shutter: object,
    shutter_open: typing.Any,
    shutter_close: typing.Any,
    shutter_wait_open: float = 0.,
    shutter_wait_close: float = 0.,
    take_dark: bool = True,
    md: dict = None,
    backoff: float = 0.,
    snake: bool = False,
) -> typing.Generator:
    """Move on a grid and do a fly scan at each point in the grid.

    For example, `fly_scan_nd([detector], motor_y, 0, 10, 11,
    motor_x, 0, 20, 21, motor_fly, 0, 5, 6,
    time_per_point=10, time_per_frame=1, shutter=shutter,
    shutter_open=1, shutter_close=0,
    shutter_wait_open=2, shutter_wait_close=5,
    move_velocity=5, take_dark=True,
    md={"task": "fly scan sample 1", backoff=0.5, snake=False})`
    means that set detector so that it will collect one image for 10 s
    one image contains 10 frames and each frame for 1 s

    for y in 0, 1, 2, ..., 10:
    for x in 0, 1, 2, ..., 20:
    move to (x, y)
    wait 5 s
    collect dark image during the movement
    open shutter
    wait 2 s
    fly scan the motor_fly from -0.5 to 5.5
    collect 6 images during the fly
    close shutter

    Parameters
    ----------
    detectors : list
        A list of detectors. The first one must be an area detector. The list shouldn't include the motors in the
        `args`.
    *args :
        patterned like (``motor1, start1, stop1, num1,``
                        ``motor2, start2, stop2, num2,``
                        ``motor3, start3, stop3, num3,`` ...
                        ``motorN, startN, stopN, numN``)

        The first motor is the "slowest", the outer loop. The last motor
        is the "fly" motor, the non-stoping scan along an axis. For all motors
        except the first motor, there is a "snake" argument: a boolean
        indicating whether to following snake-like, winding trajectory or a
        simple left-to-right trajectory.
    move_velocity : float
        The speed for the motors to move to the next grid point.
    time_per_point : float
        The time to collect one image at one point.
    time_per_frame : float
        The time to collect one frame in a image. One image contains serveral frames.
    shutter : object
        The fast shutter.
    shutter_open : typing.Any
        The value of the shutter in open state.
    shutter_close : typing.Any
        The value of the shutter in close state.
    shutter_wait_open : float, optional
        The time between the shutter open and the start of the light image collection, by default 0.
    shutter_wait_close : float, optional
        The time between the shutter close and the start of the dark image collection, by default 0.
    take_dark : bool, optional
        If true, take a dark image at the end of the fly scan, by default True
    md : dict, optional
        The dictionary of the metadata to added into the plan, by default None
    backoff : float, optional
        If non-zero, fly scan from start - backoff to end + backoff, by default 0.
    snake : bool, optional
        If true, snake the axis of the fly scan, by default False

    Yields
    -------
    Iterator[typing.Generic]
        The messages of the plan.

    Raises
    ------
    TomoPlanError
        Empty detector list.
    TomoPlanError
        Not enough motors.
    TomoPlanError
        Wrong motor positions format.
    """
    # check args
    if not detectors:
        raise TomoPlanError("dets cannot be an empty list.")
    if len(args) < 8:
        raise TomoPlanError(
            "There must be at least 8 arguments for the motors, like ``motor1, start1, end1, number1, motor2, "
            "start2, end2, number2`.")
    if len(args) % 4 != 0:
        raise TomoPlanError(
            "The arguments must be in format `motor1, start1, end1, number1, motor2, start2, end2, number2, ...`")

    # get the motors and positions
    fly_motor = args[-4]
    fly_start = args[-3]
    fly_stop = args[-2]
    fly_pixels = args[-1]
    motors = args[0:-4:4]
    starts = args[1:-4:4]
    ends = args[2:-4:4]
    nums = args[3:-4:4]
    coords = _get_motors_and_coords(starts, ends, nums)

    # configurea area detector
    ad = detectors[0]
    num_frame, acq_time, computed_dwell_time = yield from configure_area_det(
        ad, time_per_point, time_per_frame
    )

    # set up metadata
    sp = {
        "time_per_frame": acq_time,
        "num_frames": num_frame,
        "requested_exposure": time_per_point,
        "computed_exposure": computed_dwell_time,
        "type": "ct",
        "uid": str(uuid.uuid4()),
        "plan_name": "fly_scan_nd",
    }
    _md = {
        "detectors": [det.name for det in detectors],
        "plan_args": {},
        "map_size": (fly_pixels,) + nums,
        "hints": {},
        "extents": [(fly_start, fly_stop)] + [(start, end) for start, end in zip(starts, ends)],
        **{f"sp_{k}": v for k, v in sp.items()},
    }
    _md.update(md or {})
    _md["hints"].setdefault(
        "dimensions",
        [((f"start_{fly_motor.name}",), "primary")] + [((motor.name,), "primary") for motor in motors],
    )
    # soft signal to use for tracking pixel edges
    px_start = Signal(name=f"start_{fly_motor.name}", kind=Kind.normal)
    px_stop = Signal(name=f"stop_{fly_motor.name}", kind=Kind.normal)
    all_dets = detectors + [px_start, px_stop, fly_motor.velocity] + list(motors)
    # or get the gating working below.
    speed = abs(fly_stop - fly_start) / (fly_pixels * computed_dwell_time)
    # check the speed
    low, high = sorted(fly_motor.velocity.limits)
    if speed < low or speed > high:
        raise ValueError("The fly scan velocity is {}. ".format(speed) +
                         "It is out of the range of the motor speed: ({}, {}).".format(low, high))
    if move_velocity < low or move_velocity > high:
        raise ValueError("The move velocity is {}. ".format(move_velocity) +
                         "It is out of the range of the motor speed: ({}, {}).".format(low, high))

    @bpp.reset_positions_decorator([fly_motor.velocity])
    @bpp.set_run_key_decorator(f"xrd_map_{uuid.uuid4()}")
    @bpp.stage_decorator(detectors)
    @bpp.run_decorator(md=_md)
    def inner():
        _fly_start, _fly_stop = fly_start, fly_stop
        _backoff = backoff
        for coord in coords:
            # move the step motors
            yield from bps.checkpoint()
            yield from bps.mv(fly_motor.velocity, move_velocity)
            pre_fly_group = short_uid("pre_fly")
            for motor, pos in zip(motors, coord):
                yield from bps.abs_set(motor, pos, group=pre_fly_group)
            yield from bps.abs_set(
                fly_motor, _fly_start - _backoff, group=pre_fly_group
            )
            # take a dark during the movement
            if take_dark:
                yield from bps.sleep(shutter_wait_close)
                yield from dark_plan(ad)
            yield from bps.wait(group=pre_fly_group)

            # wait for the pre-fly motion to stop
            yield from bps.mv(fly_motor.velocity, speed)
            yield from bps.mv(shutter, shutter_open)
            yield from bps.sleep(shutter_wait_open)
            fly_group = short_uid("fly")
            yield from bps.abs_set(fly_motor, _fly_stop + _backoff, group=fly_group)
            for j in range(fly_pixels):

                fly_pixel_group = short_uid("fly_pixel")
                for d in detectors:
                    yield from bps.trigger(d, group=fly_pixel_group)

                # grab motor position right after we trigger
                start_pos = yield from _extarct_motor_pos(fly_motor)
                yield from bps.mv(px_start, start_pos)
                # wait for frame to finish
                yield from bps.wait(group=fly_pixel_group)

                # grab the motor position
                stop_pos = yield from _extarct_motor_pos(fly_motor)
                yield from bps.mv(px_stop, stop_pos)
                # generate the event
                yield from bps.create("primary")
                for obj in all_dets:
                    yield from bps.read(obj)
                yield from bps.save()
            yield from bps.checkpoint()
            yield from bps.mv(shutter, shutter_close)
            yield from bps.wait(group=fly_group)
            yield from bps.checkpoint()
            if snake:
                # if snaking, flip these for the next pass through
                _fly_start, _fly_stop = _fly_stop, _fly_start
                _backoff = -_backoff
        return

    return (yield from inner())


def fly_scan_nd_no_shutter(
    detectors: list,
    *args,
    move_velocity: float,
    time_per_point: float,
    time_per_frame: float,
    shutter: object,
    shutter_open: typing.Any,
    shutter_close: typing.Any,
    shutter_wait_open: float = 0.,
    shutter_wait_close: float = 0.,
    take_dark: bool = True,
    md: dict = None,
    backoff: float = 0.,
    snake: bool = False,
) -> typing.Generator:
    """Move on a grid and do a fly scan at each point in the grid.

    Parameters
    ----------
    detectors : list
        A list of detectors. The first one must be an area detector. The list shouldn't include the motors in the
        `args`.
    *args :
        patterned like (``motor1, start1, stop1, num1,``
                        ``motor2, start2, stop2, num2,``
                        ``motor3, start3, stop3, num3,`` ...
                        ``motorN, startN, stopN, numN``)

        The first motor is the "slowest", the outer loop. The last motor
        is the "fly" motor, the non-stoping scan along an axis. For all motors
        except the first motor, there is a "snake" argument: a boolean
        indicating whether to following snake-like, winding trajectory or a
        simple left-to-right trajectory.
    move_velocity : float
        The speed for the motors to move to the next grid point.
    time_per_point : float
        The time to collect one image at one point.
    time_per_frame : float
        The time to collect one frame in a image. One image contains serveral frames.
    shutter : object
        The fast shutter.
    shutter_open : typing.Any
        The value of the shutter in open state.
    shutter_close : typing.Any
        The value of the shutter in close state.
    shutter_wait_open : float, optional
        The time between the shutter open and the start of the light image collection, by default 0.
    shutter_wait_close : float, optional
        The time between the shutter close and the start of the dark image collection, by default 0.
    take_dark : bool, optional
        If true, take a dark image at the end of the fly scan, by default True
    md : dict, optional
        The dictionary of the metadata to added into the plan, by default None
    backoff : float, optional
        If non-zero, fly scan from start - backoff to end + backoff, by default 0.
    snake : bool, optional
        If true, snake the axis of the fly scan, by default False

    Yields
    -------
    Iterator[typing.Generic]
        The messages of the plan.

    Raises
    ------
    TomoPlanError
        Empty detector list.
    TomoPlanError
        Not enough motors.
    TomoPlanError
        Wrong motor positions format.
    """
    # check args
    if not detectors:
        raise TomoPlanError("dets cannot be an empty list.")
    if len(args) < 8:
        raise TomoPlanError(
            "There must be at least 8 arguments for the motors, like ``motor1, start1, end1, number1, motor2, "
            "start2, end2, number2`.")
    if len(args) % 4 != 0:
        raise TomoPlanError(
            "The arguments must be in format `motor1, start1, end1, number1, motor2, start2, end2, number2, ...`")

    # get the motors and positions
    fly_motor = args[-4]
    fly_start = args[-3]
    fly_stop = args[-2]
    fly_pixels = args[-1]
    motors = args[0:-4:4]
    starts = args[1:-4:4]
    ends = args[2:-4:4]
    nums = args[3:-4:4]
    coords = _get_motors_and_coords(starts, ends, nums)

    # configurea area detector
    ad = detectors[0]
    num_frame, acq_time, computed_dwell_time = yield from configure_area_det(
        ad, time_per_point, time_per_frame
    )

    # set up metadata
    sp = {
        "time_per_frame": acq_time,
        "num_frames": num_frame,
        "requested_exposure": time_per_point,
        "computed_exposure": computed_dwell_time,
        "type": "ct",
        "uid": str(uuid.uuid4()),
        "plan_name": "fly_scan_nd",
    }
    _md = {
        "detectors": [det.name for det in detectors],
        "plan_args": {},
        "map_size": (fly_pixels,) + nums,
        "shape": (fly_pixels,) + nums,
        "hints": {},
        "extents": [(fly_start, fly_stop)] + [(start, end) for start, end in zip(starts, ends)],
        **{f"sp_{k}": v for k, v in sp.items()},
    }
    _md.update(md or {})
    _md["hints"].setdefault(
        "dimensions",
        [((f"start_{fly_motor.name}",), "primary")] + [((motor.name,), "primary") for motor in motors],
    )
    # soft signal to use for tracking pixel edges
    px_start = Signal(name=f"start_{fly_motor.name}", kind=Kind.normal)
    px_stop = Signal(name=f"stop_{fly_motor.name}", kind=Kind.normal)
    all_dets = detectors + [px_start, px_stop, fly_motor.velocity] + list(motors)
    # or get the gating working below.
    speed = abs(fly_stop - fly_start) / (fly_pixels * computed_dwell_time)
    # check the speed
    low, high = sorted(fly_motor.velocity.limits)
    if speed < low or speed > high:
        raise ValueError("The fly scan velocity is {}. ".format(speed) +
                         "It is out of the range of the motor speed: ({}, {}).".format(low, high))
    if move_velocity < low or move_velocity > high:
        raise ValueError("The move velocity is {}. ".format(move_velocity) +
                         "It is out of the range of the motor speed: ({}, {}).".format(low, high))

    @bpp.reset_positions_decorator([fly_motor.velocity])
    @bpp.set_run_key_decorator(f"xrd_map_{uuid.uuid4()}")
    @bpp.stage_decorator(detectors)
    @bpp.run_decorator(md=_md)
    def inner():
        _fly_start, _fly_stop = fly_start, fly_stop
        _backoff = backoff
        yield from bps.mv(shutter, shutter_open)
        for coord in coords:
            # move the step motors
            yield from bps.checkpoint()
            yield from bps.mv(fly_motor.velocity, move_velocity)
            pre_fly_group = short_uid("pre_fly")
            for motor, pos in zip(motors, coord):
                yield from bps.abs_set(motor, pos, group=pre_fly_group)
            yield from bps.abs_set(
                fly_motor, _fly_start - _backoff, group=pre_fly_group
            )
            # take a dark during the movement
            if take_dark:
                yield from bps.mv(shutter, shutter_close)
                yield from bps.sleep(shutter_wait_close)
                yield from dark_plan(ad)
                yield from bps.sleep(shutter_wait_open)
                yield from bps.mv(shutter, shutter_open)
            yield from bps.wait(group=pre_fly_group)

            fly_group = short_uid("fly")
            yield from bps.mv(fly_motor.velocity, speed)
            yield from bps.abs_set(fly_motor, _fly_stop + _backoff, group=fly_group)
            for j in range(fly_pixels):

                fly_pixel_group = short_uid("fly_pixel")
                for d in detectors:
                    yield from bps.trigger(d, group=fly_pixel_group)

                # grab motor position right after we trigger
                start_pos = yield from _extarct_motor_pos(fly_motor)
                yield from bps.mv(px_start, start_pos)
                # wait for frame to finish
                yield from bps.wait(group=fly_pixel_group)

                # grab the motor position
                stop_pos = yield from _extarct_motor_pos(fly_motor)
                yield from bps.mv(px_stop, stop_pos)
                # generate the event
                yield from bps.create("primary")
                for obj in all_dets:
                    yield from bps.read(obj)
                yield from bps.save()
            yield from bps.checkpoint()
            yield from bps.wait(group=fly_group)
            yield from bps.checkpoint()
            if snake:
                # if snaking, flip these for the next pass through
                _fly_start, _fly_stop = _fly_stop, _fly_start
                _backoff = -_backoff
        yield from bps.mv(shutter, shutter_close)
        return

    return (yield from inner())


def grid_scan_nd(
    detectors: list,
    *args,
    snake: typing.Union[list, bool] = None,
    time_per_point: float,
    time_per_frame: float,
    shutter: object,
    shutter_open: typing.Any,
    shutter_close: typing.Any,
    shutter_wait_open: float = 0.,
    shutter_wait_close: float = 0.,
    take_dark: bool = True,
    md=None
) -> typing.Generator:
    """Scan over a mesh; each motor is on an independent trajectory.

    Parameters
    ----------
    detectors : list
        A list of 'readable' objects
    *args :
        patterned like (``motor1, start1, stop1, num1,``
                        ``motor2, start2, stop2, num2,``
                        ``motor3, start3, stop3, num3,`` ...
                        ``motorN, startN, stopN, numN``)

        The first motor is the "slowest", the outer loop. The last motor
        is the "fly" motor, the non-stoping scan along an axis. For all motors
        except the first motor, there is a "snake" argument: a boolean
        indicating whether to following snake-like, winding trajectory or a
        simple left-to-right trajectory.
    snake : bool, optional
        If true, snake the axis of the fly scan, by default None
    time_per_point : float
        The time to collect one image at one point.
    time_per_frame : float
        The time to collect one frame in a image. One image contains serveral frames.
    time_per_point : float
        The time to collect one image at one point.
    time_per_frame : float
        The time to collect one frame in a image. One image contains serveral frames.
    shutter : object
        The fast shutter.
    shutter_open : typing.Any
        The value of the shutter in open state.
    shutter_close : typing.Any
        The value of the shutter in close state.
    shutter_wait_open : float, optional
        The time between the shutter open and the start of the light image collection, by default 0.
    shutter_wait_close : float, optional
        The time between the shutter close and the start of the dark image collection, by default 0.
    take_dark : bool, optional
        If true, take a dark image at the end of the fly scan, by default True
    md : [type], optional
        The dictionary of the metadata to added into the plan, by default None, by default None

    Yields
    -------
    Iterator[typing.Generic]
        The messages of the plan.

    Raises
    ------
    TomoPlanError
        Empty detector list.
    TomoPlanError
        Not enough motors.
    TomoPlanError
        Wrong motor positions format.
    """
    if not md:
        md = {}
    if len(args) == 0:
        raise TomoPlanError("Missing arguments for the motors.")
    if len(args) % 4 != 0:
        raise TomoPlanError(
            "The arguments must be in format `motor1, start1, end1, number1, motor2, start2, end2, number2, ...`")
    if len(detectors) == 0:
        raise TomoPlanError("dets cannot be an empty list.")

    slow_motor = args[0]

    def _per_step(detectors, step, pos_cache):
        # start movement
        move_group = short_uid("move")
        for motor, pos in step.items():
            yield from bps.abs_set(motor, pos, group=move_group)
        # take dark if there is a movement of the slow motor
        if step.get(slow_motor) != pos_cache[slow_motor] and take_dark:
            yield from bps.mv(shutter, shutter_close)
            yield from bps.sleep(shutter_wait_close)
            yield from dark_plan(detectors[0])
            yield from bps.mv(shutter, shutter_open)
        # wait until finish and take reading
        yield from bps.wait(move_group)
        yield from bps.trigger_and_read(detectors)
        # update the cache
        for motor, pos in step.items():
            pos_cache[motor] = pos

    num_frame, time_per_frame2, time_per_point2 = yield from configure_area_det(
        detectors[0], time_per_point, time_per_frame
    )
    _md = {
        "sp_time_per_frame": time_per_frame2,
        "sp_num_frames": num_frame,
        "sp_requested_exposure": time_per_point,
        "sp_computed_exposure": time_per_point2,
        "sp_type": "ct",
        "sp_uid": str(uuid.uuid4()),
        "sp_plan_name": "grid_scan_nd",
    }
    _md.update(md)

    plan = bp.grid_scan(detectors, *args, snake_axes=snake, per_step=_per_step, md=_md)
    if not take_dark:
        plan = bpp.pchain(bps.mv(shutter, shutter_open), bps.sleep(shutter_wait_open), plan)
    plan = bpp.finalize_wrapper(plan, bps.mv(shutter, shutter_close))
    return (yield from plan)


def grid_scan_no_dark(
    detectors: list,
    *args,
    snake: typing.Union[list, bool] = None,
    time_per_point: float,
    time_per_frame: float,
    shutter: object,
    shutter_open: typing.Any,
    shutter_close: typing.Any,
    shutter_wait_open: float = 0.,
    md=None
) -> typing.Generator:
    """Scan over a mesh; each motor is on an independent trajectory.

    Parameters
    ----------
    detectors : list
        A list of 'readable' objects
    *args :
        patterned like (``motor1, start1, stop1, num1,``
                        ``motor2, start2, stop2, num2,``
                        ``motor3, start3, stop3, num3,`` ...
                        ``motorN, startN, stopN, numN``)

        The first motor is the "slowest", the outer loop. The last motor
        is the "fly" motor, the non-stoping scan along an axis. For all motors
        except the first motor, there is a "snake" argument: a boolean
        indicating whether to following snake-like, winding trajectory or a
        simple left-to-right trajectory.
    snake : bool, optional
        If true, snake the axis of the fly scan, by default None
    time_per_point : float
        The time to collect one image at one point.
    time_per_frame : float
        The time to collect one frame in a image. One image contains serveral frames.
    time_per_point : float
        The time to collect one image at one point.
    time_per_frame : float
        The time to collect one frame in a image. One image contains serveral frames.
    shutter : object
        The fast shutter.
    shutter_open : typing.Any
        The value of the shutter in open state.
    shutter_close : typing.Any
        The value of the shutter in close state.
    shutter_wait_open : float, optional
        The time between the shutter open and the start of the light image collection, by default 0.
    md : [type], optional
        The dictionary of the metadata to added into the plan, by default None, by default None

    Yields
    -------
    Iterator[typing.Generic]
        The messages of the plan.

    Raises
    ------
    TomoPlanError
        Empty detector list.
    TomoPlanError
        Not enough motors.
    TomoPlanError
        Wrong motor positions format.
    """
    if not md:
        md = {}
    if len(args) == 0:
        raise TomoPlanError("Missing arguments for the motors.")
    if len(args) % 4 != 0:
        raise TomoPlanError(
            "The arguments must be in format `motor1, start1, end1, number1, motor2, start2, end2, number2, ...`")
    if len(detectors) == 0:
        raise TomoPlanError("dets cannot be an empty list.")

    num_frame, time_per_frame2, time_per_point2 = yield from configure_area_det(
        detectors[0], time_per_point, time_per_frame
    )
    _md = {
        "sp_time_per_frame": time_per_frame2,
        "sp_num_frames": num_frame,
        "sp_requested_exposure": time_per_point,
        "sp_computed_exposure": time_per_point2,
        "sp_type": "ct",
        "sp_uid": str(uuid.uuid4()),
        "sp_plan_name": "grid_scan_nd",
    }
    _md.update(md)

    plan = bp.grid_scan(detectors, *args, snake_axes=snake, md=_md)
    plan = bpp.pchain(bps.mv(shutter, shutter_open), bps.sleep(shutter_wait_open), plan)
    plan = bpp.finalize_wrapper(plan, bps.mv(shutter, shutter_close))
    return (yield from plan)


def loop_until(motor: ophyd.Device, left: float, right: float, t: float) -> typing.Generic:
    """Move motor from left to right and right to left repeatedly until t seconds pass"""
    t0 = time.time()
    while time.time() - t0 < t:
        yield from bps.mv(motor, left)
        yield from bps.mv(motor, right)
    return
