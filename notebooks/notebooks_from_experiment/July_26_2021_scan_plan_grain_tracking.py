"""
1. Further detector-sample distance to get more diffraction spots on the detector
2. More intense beam than last time
3. Slit beam vs. focus beam: slit beam is slightly better
4. Filters and exposure time need to be determined at the beginning of the experiment
5. x, y and phi range also need to be determined at the beginning of the experiment
6. motor name need to be changed and match the real device name at the beamline
"""

import crystalmapping.plans as plans

# set motor names and shutter
motor_x, motor_y, motor_phi = "TBD", "TBD", "TBD"
shutter = "TBD"

# grid parameters
start_x, stop_x = "TBD", "TBD"
start_y, stop_y = "TBD", "TBD"
start_phi, stop_phi = "TBD", "TBD"

step_size_x_step_grid = 0.25
step_size_y_step_grid = 0.25

npt_x_grid = int((stop_x - start_x) / step_size_x_step_grid) + 1
npt_y_grid = int((stop_y - start_y) / step_size_y_step_grid) + 1

# single point rocking curve parameters
x_pos, y_pos = "TBD", "TBD"
phi_start, phi_end = "TBD", "TBD"
step_size_phi_single_point_rocking = 0.001
npt_phi_single_point_rocking = int((phi_end - phi_start) / step_size_phi_single_point_rocking) + 1

# coarse rocking curve parameters
step_size_x_coarse_rocking = 1
step_size_y_coarse_rocking = 2
step_size_phi_coarse_rocking = 0.01

npt_x_coarse_rocking = int((stop_x - start_x) / step_size_x_coarse_rocking) + 1
npt_y_coarse_rocking = int((stop_y - start_y) / step_size_y_coarse_rocking) + 1
npt_phi_coarse_rocking = int((stop_phi - start_phi) / step_size_phi_coarse_rocking) + 1

###########################################################################################################################

# step grid scan (snake scan)
# motor name need to be changed and match the real device name at the beamline
plan_step_grid = plans.grid_scan_nd(
    [detector],
    motor_y, start_y, stop_y, npt_y_grid,
    motor_x, start_x, stop_x, npt_x_grid,
    snake=True,
    time_per_point="TBD",
    time_per_frame="TBD",
    shutter=shutter,
    shutter_open="open",
    shutter_close="close",
    shutter_wait_open=1.0,
    shutter_wait_close=5.0,
    take_dark=True,
    md={"sample code": "XX-XXXX",
        "sample coposition": "Titanium dioxide(rutile)",
        "scan type": "step grid scan"
        }
)

# precession step grid scan (snake scan)
# motor name need to be changed and match the real device name at the beamline

# before run this code, open another terminal for precession
plan_precession_step_grid = plans.grid_scan_nd(
    [detector],
    motor_y, start_y, stop_y, npt_y_grid,
    motor_x, start_x, stop_x, npt_x_grid,
    snake=True,
    time_per_point="TBD",
    time_per_frame="TBD",
    shutter=shutter,
    shutter_open="open",
    shutter_close="close",
    shutter_wait_open=1.0,
    shutter_wait_close=5.0,
    take_dark=True,
    md={"sample code": "XX-XXXX",
        "sample coposition": "Titanium dioxide(rutile)",
        "scan type": "step grid scan"
        }
)

# single point rocking curve

# before run this code, move the x, y motor to the correct position

plan_precession_step_grid = plans.grid_scan_nd(
    [detector],
    motor_phi, phi_start, phi_end, npt_phi_single_point_rocking,
    time_per_point="TBD",
    time_per_frame="TBD",
    shutter=shutter,
    shutter_open="open",
    shutter_close="close",
    shutter_wait_open=1.0,
    shutter_wait_close=5.0,
    take_dark=True,
    md={"sample code": "XX-XXXX",
        "sample coposition": "Titanium dioxide(rutile)",
        "scan type": "single point rocking curve"
        }
)

# coarse rocking curve at 0
# motor name need to be changed and match the real device name at the beamline
plan_coarse_rocking_0 = plans.grid_scan_nd(
    [detector],
    motor_y, start_y, stop_y, npt_y_coarse_rocking,
    motor_x, start_x, stop_x, npt_x_coarse_rocking,
    motor_phi, start_phi, stop_phi, npt_phi_coarse_rocking,
    snake=True,
    time_per_point="TBD",
    time_per_frame="TBD",
    shutter=shutter,
    shutter_open="open",
    shutter_close="close",
    shutter_wait_open=1.0,
    shutter_wait_close=5.0,
    take_dark=True,
    md={"sample code": "XX-XXXX",
        "sample coposition": "Titanium dioxide(rutile)",
        "scan type": "coarse rocking curve at 0"
        }
)

# coarse rocking curve at 45
# motor name need to be changed and match the real device name at the beamline

# before run this code, rotate the crystal 45 degree from the original position

plan_coarse_rocking_45 = plans.grid_scan_nd(
    [detector],
    motor_y, start_y, stop_y, npt_y_coarse_rocking,
    motor_x, start_x, stop_x, npt_x_coarse_rocking,
    motor_phi, start_phi, stop_phi, npt_phi_coarse_rocking,
    snake=True,
    time_per_point="TBD",
    time_per_frame="TBD",
    shutter=shutter,
    shutter_open="open",
    shutter_close="close",
    shutter_wait_open=1.0,
    shutter_wait_close=5.0,
    take_dark=True,
    md={"sample code": "XX-XXXX",
        "sample coposition": "Titanium dioxide(rutile)",
        "scan type": "coarse rocking curve at 45"
        }
)

# coarse rocking curve at 90
# motor name need to be changed and match the real device name at the beamline

# before run this code, rotate the crystal for another 45 degree

plan_coarse_rocking_90 = plans.grid_scan_nd(
    [detector],
    motor_y, start_y, stop_y, npt_y_coarse_rocking,
    motor_x, start_x, stop_x, npt_x_coarse_rocking,
    motor_phi, start_phi, stop_phi, npt_phi_coarse_rocking,
    snake=True,
    time_per_point="TBD",
    time_per_frame="TBD",
    shutter=shutter,
    shutter_open="open",
    shutter_close="close",
    shutter_wait_open=1.0,
    shutter_wait_close=5.0,
    take_dark=True,
    md={"sample code": "XX-XXXX",
        "sample coposition": "Titanium dioxide(rutile)",
        "scan type": "coarse rocking curve at 90"
        }
)

# 2D fly grid scan
# motor name need to be changed and match the real device name at the beamline
plan_2D_grid_fly = plans.fly_scan_nd(
    [detector],
    motor_y, start_y, stop_y, npt_y_grid,
    motor_x, start_x, stop_x, npt_x_grid,
    move_velocity="TBD",
    time_per_point="TBD",
    time_per_frame='TBD',
    shutter=shutter,
    shutter_open="open",
    shutter_close="close",
    shutter_wait_open=1.0,
    shutter_wait_close=5.0,
    take_dark=True,
    md={"sample code": "XX-XXXX",
        "sample coposition": "Titanium dioxide(rutile)",
        "scan type": "2D fly grid scan"
        }
)

# 3D fly grid scan (fly step grid)
# motor name need to be changed and match the real device name at the beamline
plan_3D_fly_grid_rocking = plans.fly_scan_nd(
    [detector],
    motor_phi, start_phi, stop_phi, npt_phi_coarse_rocking,
    motor_y, start_y, stop_y, npt_y_grid,
    motor_x, start_x, stop_x, npt_x_grid,
    move_velocity="TBD",
    time_per_point="TBD",
    time_per_frame='TBD',
    shutter=shutter,
    shutter_open="open",
    shutter_close="close",
    shutter_wait_open=1.0,
    shutter_wait_close=5.0,
    take_dark=True,
    md={"sample code": "XX-XXXX",
        "sample coposition": "Titanium dioxide(rutile)",
        "scan type": "3D fly grid scan (fly step grid)"
        }
)

# 3D fly grid scan (fly grid rocking curve)
# motor name need to be changed and match the real device name at the beamline
plan_3D_fly_grid_rocking = plans.fly_scan_nd(
    [detector],
    motor_y, start_y, stop_y, npt_y_grid,
    motor_x, start_x, stop_x, npt_x_grid,
    motor_phi, start_phi, stop_phi, npt_phi_coarse_rocking,
    move_velocity="TBD",
    time_per_point="TBD",
    time_per_frame='TBD',
    shutter=shutter,
    shutter_open="open",
    shutter_close="close",
    shutter_wait_open=1.0,
    shutter_wait_close=5.0,
    take_dark=True,
    md={"sample code": "XX-XXXX",
        "sample coposition": "Titanium dioxide(rutile)",
        "scan type": "3D fly grid scan (fly grid rocking curve)"
        }
)
