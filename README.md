# Dwei
## A gesture-controlled, two-wheeled self-balancing robot.

This repository contains the Python-based physics simulation and Linear Quadratic Regulator (LQR) control environment. The primary goal of this simulation is to mathematically model the system's dynamics, derive the optimal gain matrix (K), and verify stability—effectively speeding up hardware development by eliminating blind trial-and-error tuning.

### State-Space Representation

The LQR controller relies on a linearized state-space model (x˙=Ax+Bu) to calculate the restorative motor forces. The chosen state vector for the feedback loop is:

```
```
x=[θ,θ˙,v,ψ˙]T
```

    θ: Body tilt angle (Pitch) with respect to the vertical normal.
    θ˙: Pitch rate.
    v: Linear velocity.
    ψ˙: Yaw rate.

### Sensor Fusion & Hardware Implementation


The physical robot is driven by an STM32F401 microcontroller. To close the control loop, the system cannot measure these states directly. Instead, it estimates the state vector by passing raw accelerometer and gyroscope data from an onboard MPU6050 through an Extended Kalman Filter (EKF).
