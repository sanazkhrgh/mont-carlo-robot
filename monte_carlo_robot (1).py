
# Monte Carlo Localization (MCL) - Robot Localization

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import Image
from google.colab import files

# -----------------------------
# 1. World and landmarks
# -----------------------------
world_size = 100.0 # 100x100 square world
landmarks = [(20, 20), (80, 80), (20, 80), (80, 20)] # known landmarks

# -----------------------------
# 2. Robot class
# -----------------------------
class Robot:
  def __init__(self):
     self.x = np.random.uniform(0, world_size)
     self.y = np.random.uniform(0, world_size)
     self.orientation = np.random.uniform(0, 2*np.pi)
     self.forward_noise = 0.5
     self.turn_noise = 0.1
     self.sense_noise = 2.0

  def set(self, new_x, new_y, new_orientation):
    self.x = float(new_x)
    self.y = float(new_y)
    self.orientation = float(new_orientation)

  def set_noise(self, f_noise, t_noise, s_noise):
    self.forward_noise = f_noise
    self.turn_noise = t_noise
    self.sense_noise = s_noise

  def move(self, turn, forward):
# turn and add noise
   orientation = self.orientation + float(turn) + np.random.normal(0, self.turn_noise)
   orientation %= 2*np.pi
# move forward with noise
   dist = float(forward) + np.random.normal(0, self.forward_noise)
   x = self.x + (np.cos(orientation) * dist)
   y = self.y + (np.sin(orientation) * dist)
   x %= world_size
   y %= world_size
   res = Robot()
   res.set(x, y, orientation)
   res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
   return res

  def sense(self):
# measure distances to landmarks with noise
   Z = []
   for lx, ly in landmarks:
     dist = np.sqrt((self.x - lx)**2 + (self.y - ly)**2)
     dist += np.random.normal(0, self.sense_noise)
     Z.append(dist)
   return Z

  def measurement_prob(self, measurement):
# calculate measurement probability
    prob = 1.0
    for i, (lx, ly) in enumerate(landmarks):
      dist = np.sqrt((self.x - lx)**2 + (self.y - ly)**2)
      prob *= np.exp(- ((dist - measurement[i])**2) / (2 * self.sense_noise**2)) /               np.sqrt(2 * np.pi * self.sense_noise**2)
    return prob

# -----------------------------
# 3. Monte Carlo Localization
# -----------------------------
N = 500 # number of particles
T = 30 # time steps

myrobot = Robot()
particles = [Robot() for _ in range(N)]

robot_path = []
particles_history = []

for t in range(T):
# move real robot
   myrobot = myrobot.move(0.1, 5)
   Z = myrobot.sense()
   robot_path.append((myrobot.x, myrobot.y))
# move particles
   particles = [p.move(0.1, 5) for p in particles]

# calculate weights
   w = [p.measurement_prob(Z) for p in particles]

# resampling using resampling wheel
   new_particles = []
   index = int(np.random.random() * N)
   beta = 0.0
   mw = max(w)
   for i in range(N):
     beta += np.random.random() * 2.0 * mw
     while beta > w[index]:
       beta -= w[index]
       index = (index + 1) % N
     new_particles.append(particles[index])
   particles = new_particles

   particles_history.append([(p.x, p.y) for p in particles])

# -----------------------------
# 4. Animation
# -----------------------------
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0, world_size)
ax.set_ylim(0, world_size)
ax.scatter(*zip(*landmarks), marker="*", s=200, c="red", label="Landmarks")
robot_dot, = ax.plot([], [], "bo", markersize=10, label="Robot")
particles_dot, = ax.plot([], [], "go", alpha=0.3, linestyle="", marker="o", label="Particles")
ax.legend()

def animate(i):
   rx, ry = robot_path[i]
   robot_dot.set_data([rx], [ry]) # wrap in list
   px, py = zip(*particles_history[i])
   particles_dot.set_data(px, py)
   return robot_dot, particles_dot

ani = animation.FuncAnimation(fig, animate, frames=T, interval=300, blit=True)

# -----------------------------
# Save animation as GIF
ani.save("/content/mcl.gif", writer="pillow")
print("Saved: /content/mcl.gif")

# Show animation inside Colab (interactive)
from IPython.display import HTML
HTML(ani.to_jshtml())

