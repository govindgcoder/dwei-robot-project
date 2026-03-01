#imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.linalg as linalg
from matplotlib.widgets import Slider, Button

#constants
g = 9.81
mass_body = 0.6 
mass_wheel = 0.03 
L_seperation = 0.06 
radius_wheel = 0.0325 
Ip = (1/12) * mass_body * L_seperation**2 
Iw = 0.5 * mass_wheel * radius_wheel**2 
R_motor = 6.0 
Kt = 0.015 
Ke = 0.015 

W_track = 0.2 
# assume to be a rect
Iz = (1/12) * mass_body * (W_track**2 + L_seperation**2) 



x_ref = np.array([[0],[0],[0],[0]])

# mass matrix
E_mat = np.array(
    [
        [1,0,0,0],
        [0, Ip+mass_body*L_seperation**2,mass_body*L_seperation,0],
        [0,mass_body*L_seperation,mass_body+2*mass_wheel+2*Iw/radius_wheel**2,0],
        [0,0,0,1]
    ])
# natural physics
A_rhs = np.array(
    [
        [0,1,0,0],
        [mass_body*g*L_seperation,0,2*Ke*Kt/(R_motor*radius_wheel),0],
        [0,0,-2*Ke*Kt/(R_motor*radius_wheel**2),0],
        [0,0,0,-(Kt * Ke * W_track**2) / (2 * R_motor * radius_wheel**2 * Iz)]
    ])
# motor authority matrix
B_rhs = np.array(
    [
        [0,0],
        [-2*Kt/R_motor,0],
        [2*Kt/(R_motor*radius_wheel),0],
        [0,(Kt * W_track) / (2 * R_motor * radius_wheel * Iz)]
    ])

A = np.linalg.inv(E_mat) @ A_rhs
B = np.linalg.inv(E_mat) @ B_rhs

# yaw physical reactions
B_yaw_authority = (Kt * W_track) / (2 * R_motor * radius_wheel * Iz)


import scipy.linalg as la

# Q matrix: Penalties for [theta, theta_dot, velocity, yaw_rate]
Q = np.diag([100.0, 1.0, 75.0,10]) 

# R matrix: Penalty for actuator effort (V_avg, V_diff)
R = np.diag([0.5,0.5])

P = la.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

print("K matrix:", K)

dt = 0.01  # Simulate at 100Hz (same as your STM32 loop)
time_steps = 10000

# State vector: [theta, theta_dot, v, yaw_rate] (column vector)
x = np.array([[0.3], [0.0], [0.0], [0.0]])

# Lists to store data for plotting later
history_theta = []
history_v = []
history_u = []


dt = 0.02 # 50Hz simulation 
x_state = np.array([[0.3], [0.0], [0.0], [0.0]]) # Start tilted

x_ref = np.array([[0.0], [0.0], [0.0], [0.0]])   # Target state

pos_local = 0.0     
pos_global_x = 0.0  
pos_global_y = 0.0   
yaw_angle = 0.0      

fig = plt.figure(figsize=(12, 8))
plt.subplots_adjust(bottom=0.25) # Make room for sliders

ax_side = fig.add_subplot(221, aspect='equal')
ax_side.set_ylim(-0.05, 0.3)
ax_side.set_title("Side")
ax_side.grid(True)
side_ground, = ax_side.plot([-10, 10], [0, 0], 'k-', lw=2)
side_body, = ax_side.plot([], [], 'r-', lw=4)
side_wheel = plt.Circle((0, radius_wheel), radius_wheel, color='blue', fill=False, lw=3)
ax_side.add_patch(side_wheel)

ax_front = fig.add_subplot(222, aspect='equal')
ax_front.set_xlim(-0.2, 0.2)
ax_front.set_ylim(-0.05, 0.3)

ax_front.set_title("Front")
ax_front.grid(True)
ax_front.plot([-1, 1], [0, 0], 'k-', lw=2) # Ground
front_body, = ax_front.plot([], [], 'r-', lw=6)
front_axle, = ax_front.plot([], [], 'gray', lw=3)
wheel_l = plt.Circle((-W_track/2, radius_wheel), radius_wheel, color='blue', fill=False, lw=3)
wheel_r = plt.Circle((W_track/2, radius_wheel), radius_wheel, color='blue', fill=False, lw=3)
ax_front.add_patch(wheel_l)
ax_front.add_patch(wheel_r)

ax_top = fig.add_subplot(212, aspect='equal')
ax_top.set_xlim(-2, 2)
ax_top.set_ylim(-2, 2)

ax_top.set_title("Top")
ax_top.grid(True)
top_body, = ax_top.plot([], [], 'r-', lw=4)
top_axle, = ax_top.plot([], [], 'blue', lw=4)
top_trail, = ax_top.plot([], [], 'gray', alpha=0.5, linestyle='--')
trail_x, trail_y = [], []

ax_vel = plt.axes([0.15, 0.12, 0.65, 0.03])
ax_yaw = plt.axes([0.15, 0.06, 0.65, 0.03])
ax_btn = plt.axes([0.85, 0.06, 0.1, 0.09])



slider_vel = Slider(ax_vel, 'Fwd/Rev (m/s)', -1.0, 1.0, valinit=0.0)
slider_yaw = Slider(ax_yaw, 'Left/Right (rad/s)', -3.0, 3.0, valinit=0.0)
btn_idle = Button(ax_btn, 'Idle')

def update_targets(val):
    x_ref[2, 0] = slider_vel.val
    x_ref[3, 0] = slider_yaw.val


def reset_idle(event):
    slider_vel.reset()
    slider_yaw.reset()


slider_vel.on_changed(update_targets)
slider_yaw.on_changed(update_targets)
btn_idle.on_clicked(reset_idle)

# --- The Physics Engine & Drawing Loop ---
def update(frame):
    global x_state, pos_local, pos_global_x, pos_global_y, yaw_angle
    
    # 1. Physics Integration
    u = -K @ (x_state - x_ref)
    u = np.clip(u, -12, 12)
    dx = A @ x_state + B @ u
    x_state = x_state + dx * dt
    
    theta = x_state[0, 0]
    v = x_state[2, 0]
    yaw_rate = x_state[3, 0]
    
    # 2. Update Position Odometry
    pos_local += v * dt
    yaw_angle += yaw_rate * dt
    pos_global_x += v * np.cos(yaw_angle) * dt
    pos_global_y += v * np.sin(yaw_angle) * dt
    
    trail_x.append(pos_global_x)
    trail_y.append(pos_global_y)
    # Keep trail manageable
    if len(trail_x) > 500: 

        trail_x.pop(0)
        trail_y.pop(0)
    
    # 3. Draw Side View
    ax_side.set_xlim(pos_local - 0.3, pos_local + 0.3)
    side_wheel.set_center((pos_local, radius_wheel))
    top_x = pos_local + (2 * L_seperation) * np.sin(theta)
    top_y = radius_wheel + (2 * L_seperation) * np.cos(theta)
    side_body.set_data([pos_local, top_x], [radius_wheel, top_y])
    
    # 4. Draw Front View
    # As it tilts forward, it looks "shorter" from the front
    front_height = radius_wheel + (2 * L_seperation) * np.cos(theta)
    front_body.set_data([0, 0], [radius_wheel, front_height])
    front_axle.set_data([-W_track/2, W_track/2], [radius_wheel, radius_wheel])
    
    # 5. Draw Top View
    # Camera follows the robot globally
    ax_top.set_xlim(pos_global_x - 1.0, pos_global_x + 1.0)
    ax_top.set_ylim(pos_global_y - 1.0, pos_global_y + 1.0)
    
    # Calculate axle points based on yaw
    axle_lx = pos_global_x - (W_track/2) * np.sin(yaw_angle)
    axle_ly = pos_global_y + (W_track/2) * np.cos(yaw_angle)
    axle_rx = pos_global_x + (W_track/2) * np.sin(yaw_angle)
    axle_ry = pos_global_y - (W_track/2) * np.cos(yaw_angle)
    
    # Calculate front body pointer
    nose_x = pos_global_x + (L_seperation) * np.cos(yaw_angle)
    nose_y = pos_global_y + (L_seperation) * np.sin(yaw_angle)
    
    top_axle.set_data([axle_lx, axle_rx], [axle_ly, axle_ry])
    top_body.set_data([pos_global_x, nose_x], [pos_global_y, nose_y])
    top_trail.set_data(trail_x, trail_y)
    
    return side_body, side_wheel, front_body, top_axle, top_body, top_trail


ani = animation.FuncAnimation(fig, update, interval=dt*1000, blit=False, cache_frame_data=False)

plt.show()
