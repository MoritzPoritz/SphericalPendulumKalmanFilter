"""
Run via python path/to/spherical_pendulum_sim.py

Simulation explained in this video:

https://youtu.be/p0waArLc0rA
"""

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


Y_INIT = [
    np.pi / 2.5,  # theta
    0,  # phi
    0,  # theta'
    1 * np.pi / 4.2,  # phi'
]
# Nice divisors are divisors (2.5,4.2), (4, 6), (5, 2) and (1.1, inf)
DOF = len(Y_INIT) // 2

GRAVITY_ACC = 9.80665  # [m/s^2]
PENDULUM_LENGTH = 3 # [m] # used in integration scheme as well as for plots
DT = 1 / 30  # Note: Euler-Lagrangie itself doesn't tell you what a good DT is


def _inner(vs, ws):

    assert len(vs) == len(ws)

    terms = (w * v for w, v in zip(vs, ws))

    return sum(terms)


def weighted_mean(weights, values):

    return _inner(weights, values) / sum(weights)


def transpose(points):
    if not points:
        return []
    dim = len(points[0])
    return [[point[i] for point in points] for i in range(dim)]


def angles_to_unit_sphere(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return [x, y, z]


def integrate(f, dt, y_init, t_init=0, t_fin=float('inf')):
    # Solves y'(t) = f(t, y) for a seqeunce of ys.

    dy = _dy_rk4_method(f, dt)

    y = y_init
    t = t_init

    while t < t_fin:
        yield y

        y += dy(t, y)
        t += dt


def _dy_rk4_method(f, dt):
    """
    See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    :param f: Function determining the differential equation
    :param dt: Time delta
    """

    def f_(t, y):
        return np.array(f(t, y)) # Guarantee possibility of vector addition

    def dy(t, y):
        k1 = f_(t, y)
        k2 = f_(t + dt / 2, y + k1 * dt / 2)
        k3 = f_(t + dt / 2, y + k2 * dt / 2)
        k4 = f_(t + dt, y + k3 * dt)
        WEIGHTS = [1, 2, 2, 1]

        f_rk4 = weighted_mean(WEIGHTS, [k1, k2, k3, k4])

        return f_rk4 * dt

    return dy


def _pendulum_in_physical_space(stream__hamiltonian, stream__lagrangian):
    """
    Note: angles_to_unit_sphere returns point in standard sphere coordinates.
    The model assums theta=0 corresponds to lowest potential energy,
    so we need to flip the z-axis.
    
    Only one stream is really required, the second is optional.
    """

    for yl, yh in zip(stream__lagrangian(), stream__hamiltonian()):

        # Sanity check that Lagrangian and Hamiltonian stream give same results.
        # This is optional.
        EPSILON = 10**-3 # This is the best it seems we can do
        delta_theta = abs(yl[0] - yh[0]) > EPSILON
        delta_phi = abs(yl[1] - yh[1]) > EPSILON
        if delta_theta or delta_phi:
            print(f"Attention! Lagrangian and Hamiltonian computation differ by more than {EPSILON}.")

        q = yl[:DOF] # Project out FIRST two components

        sphere_point = angles_to_unit_sphere(*q)
        sphere_point[2] *= -1 # flip z-axis
        point = [PENDULUM_LENGTH * p for p in sphere_point]
        yield point


def _pendulum_in_configuration_space__lagrangian():
    """
    For the pendulum mechanics, see
    https://en.wikipedia.org/wiki/Spherical_pendulum
    See also
    https://en.wikipedia.org/wiki/D%27Alembert%27s_principle
    https://en.wikipedia.org/wiki/Lagrangian_mechanics

    Integrates r''(t)==F for the pendulum with phase space.
    See Wikipedia link for pendulum above.
    Uses generalized coordinates (2 dim)
        q = (theta, phi).
    I.e. solve
        q'' = g(q, q')

    Force norm in physical space:
        |F| = m g.
    Height h(theta) = l * (1 - cos(theta))
    Potential in configuration space:
        V(theta) = |F| * h(theta)
        V(0) = 0
        V(pi/2) = |F| * l
    Lagrangian:
    L = T - V

    Approach:
    Let y = (q, q') (4 dim) and reduce the second order ODE q'' = g(y) to first
    order ODE y'(t) = f(t, y) via f(t, y) = (second(y), g(y)).
    """

    def f(_t, y):
        #q = y[:DOF] # Project out FIRST two components # uneeded here, since we define g in terms of y
        dq = list(y)[DOF:] # Project out LAST two components
        dq2 = g(y)
        return dq + dq2

    def g(y):
        theta, _phi, d_theta, d_phi = y
        c, s, t = np.cos(theta), np.sin(theta), np.tan(theta)

        d2_theta = (d_phi**2 * c - GRAVITY_ACC / PENDULUM_LENGTH) * s
        d2_phi = -2 * d_theta * d_phi / t
        if (t == 0):
            print ('t is 0')

        return [d2_theta, d2_phi]

    return integrate(f, DT, Y_INIT)


def _pendulum_in_configuration_space__hamiltonian():
    """
    https://en.wikipedia.org/wiki/Spherical_pendulum#Hamiltonian_mechanics
    """

    def f(_t, y):
        theta, phi, p_theta, p_phi = y
        c, s, t = np.cos(theta), np.sin(theta), np.tan(theta)

        l2 = PENDULUM_LENGTH**2
        ls2 = s**2 * l2

        d_theta = p_theta / l2
        d_phi = p_phi / (l2 * s**2)
        d_p_theta = p_phi**2 / (l2 * s**2 * t) - GRAVITY_ACC * PENDULUM_LENGTH * s
        d_p_phi = 0

        return [d_theta, d_phi, d_p_theta, d_p_phi]

    def y_lagrangian_to_hamiltonian(y_lagrangian):
        theta, phi, d_theta, d_phi = y_lagrangian
        c, s, t = np.cos(theta), np.sin(theta), np.tan(theta)

        l2 = PENDULUM_LENGTH**2

        p_theta = l2 * d_theta
        p_phi = s**2 * l2 * d_phi

        return [theta, phi, p_theta, p_phi]

    y_init = y_lagrangian_to_hamiltonian(Y_INIT)

    return integrate(f, DT, y_init)


def _plot_pendulum(ax, points):
    BOX_SIZE = 1.2 * PENDULUM_LENGTH
    BOWL_RESOLUTION = 24

    # Reset axes
    ax.cla()
    ax.set_xlim(-BOX_SIZE, BOX_SIZE)
    ax.set_ylim(-BOX_SIZE, BOX_SIZE)
    ax.set_zlim(-BOX_SIZE, BOX_SIZE)
 
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Plot bowl
    us = np.linspace(0, 2 * np.pi, BOWL_RESOLUTION)
    vs = np.linspace(0, np.pi / 3, BOWL_RESOLUTION)
    xs = np.outer(np.cos(us), np.sin(vs))
    ys = np.outer(np.sin(us), np.sin(vs))
    zs = np.outer(np.ones(np.size(us)), np.cos(vs))
    coords = [PENDULUM_LENGTH * vs for vs in [xs, ys, -zs]] # Note: scaled and inverted z-axis
    ax.plot_surface(*coords, linewidth=0, antialiased=False, cmap="coolwarm", alpha=.12)

    # Plot lines and points
    ax.plot3D(*transpose([[0, 0, BOX_SIZE / 2], [0, 0, 0]]), 'black', linewidth=1)
    ax.plot3D(*transpose([[0, 0, 0], points[-1]]), 'black', linewidth=1)
    ax.scatter3D(0, 0, 0, s=10, c="blue")
    ax.scatter3D(*transpose([points[-1]]), s=80, c="blue")
    ax.scatter3D(*transpose(points), s=1, c="green")


class PlotStream:
    __FPS = 60  # s
    __INTERVAL = 1000 / __FPS

    def __init__(self, stream):
        self.__fig = plt.figure(figsize=(6, 6))
        self.__ax = self.__fig.gca(projection='3d')
        self.__stream = stream
        self.__past_points = []

    def run(self):
        """
        Warning: Render loops might work different on different OS's and so
        this might need different arguments and a return value for __next_frame
        """
        _animation = matplotlib.animation.FuncAnimation(self.__fig,
            self.__next_frame, interval=self.__INTERVAL) # blit=True

        plt.show()

    def __next_frame(self, i):
        next_point = next(self.__stream)
        self.__past_points.append(next_point)
        _plot_pendulum(self.__ax, self.__past_points)
        #print(i, next_point) # Uncomment to print stream of pendulum points
        # return self.__ax,
        #plt.stop(0.03)


if __name__ == '__main__':
    stream__hamiltonian = _pendulum_in_configuration_space__hamiltonian
    stream__lagrangian = _pendulum_in_configuration_space__lagrangian
    stream = _pendulum_in_physical_space(stream__hamiltonian, stream__lagrangian)
    
    PlotStream(stream).run()