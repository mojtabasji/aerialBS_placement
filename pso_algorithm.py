import numpy as np
import timeit
import copy
from updates import inc_counter
from network import calc_R, objective_function, objective_function_z


class PSO(object):
    """
    Class implementing PSO algorithm.
    """

    def __init__(self, fit_func_data, init_pos, n_particles, c0, c1, w, limit, alg):
        """
        Initialize the key variables.
        Args:
          func (function): the fitness function to optimize.
          init_pos (array-like): the initial position to kick off the
                                 optimization process.
          n_particles (int): the number of particles of the swarm.
        """
        self.c0 = c0
        self.c1 = c1
        self.w = w
        self.limit = limit
        self.fit_func_data = fit_func_data
        self.alg = alg
        self.n_particles = n_particles
        self.init_pos = np.array(init_pos)
        self.particle_dim = len(init_pos)
        # Initialize particle positions using a uniform distribution
        self.particles_pos = (
            np.random.uniform(size=(n_particles, self.particle_dim)) * self.init_pos
        )
        # Initialize particle velocities using a uniform distribution
        self.velocities = np.random.uniform(size=(n_particles, self.particle_dim))

        # Initialize the best positions
        self.g_best = init_pos
        self.p_best = self.particles_pos

    def func(self, new_value, return_z=False):
        if self.alg == "PROB":
            R = calc_R(
                self.fit_func_data["bss"],
                self.fit_func_data["association_array"],
                np.array(new_value),
                self.fit_func_data["min_pt"],
            )
        elif self.alg == "UPAS":
            mixed_bss = np.concatenate(
                (self.fit_func_data["ground_bss"], np.array(new_value).reshape(-1, 2)),
                axis=0,
            )
            R = calc_R(
                mixed_bss,
                self.fit_func_data["association_array"],
                self.fit_func_data["bss_weights"],
                self.fit_func_data["min_pt"],
            )
        if return_z:
            return objective_function(R), objective_function_z(R)
        return objective_function(R)

    def update_position(self, x, v):
        """
        Update particle position.
        Args:
          x (array-like): particle current position.
          v (array-like): particle current velocity.
        Returns:
          The updated position (array-like).
        """
        x = np.array(x)
        v = np.array(v)
        new_x = x + v

        new_x = np.clip(new_x, self.limit[0], self.limit[1])
        return new_x

        # c0=0.5, c1=1.5, w=0.75

    def update_velocity(self, x, v, p_best, g_best):
        """
        Update particle velocity.
        Args:
          x (array-like): particle current position.
          v (array-like): particle current velocity.
          p_best (array-like): the best position found so far for a particle.
          g_best (array-like): the best position regarding
                               all the particles found so far.
          c0 (float): the cognitive scaling constant.
          c1 (float): the social scaling constant.
          w (float): the inertia weight
        Returns:
          The updated velocity (array-like).
        """
        x = np.array(x)
        v = np.array(v)
        assert x.shape == v.shape, "Position and velocity must have same shape"
        # a random number between 0 and 1.
        r1 = np.random.uniform()
        p_best = np.array(p_best)
        g_best = np.array(g_best)

        new_v = self.w * v + self.c0 * r1 * (p_best - x) + self.c1 * r1 * (g_best - x)
        return new_v

    def optimize(self, maxiter=20):
        """
        Run the PSO optimization process untill the stoping criteria is met.
        Case for minimization. The aim is to minimize the cost function.
        Args:
            maxiter (int): the maximum number of iterations before stopping
                           the optimization.
        Returns:
            The best solution found (array-like).
        """
        old_obf = 0
        number_equal = 1
        for ite in range(maxiter):
            start1 = timeit.default_timer()
            sum_time = 0
            my_formatted_list = ["%.3f" % elem for elem in self.g_best]
            for i in range(self.n_particles):
                x = copy.copy(self.particles_pos[i])
                v = copy.copy(self.velocities[i])
                p_best = copy.copy(self.p_best[i])
                self.velocities[i] = self.update_velocity(x, v, p_best, self.g_best)
                self.particles_pos[i] = self.update_position(x, v)
                start2 = timeit.default_timer()
                # Update the best position for particle i
                if self.func(self.particles_pos[i]) > self.func(p_best):
                    self.p_best[i] = copy.copy(self.particles_pos[i])
                # Update the best position overall
                if self.func(self.particles_pos[i]) > self.func(self.g_best):
                    self.g_best = copy.copy(self.particles_pos[i])
                end2 = timeit.default_timer()
                sum_time += end2 - start2
            end1 = timeit.default_timer()
            new_obf, obf_z = self.func(self.g_best, return_z=True)
            print(
                "%.3f  %s %d" % (new_obf, my_formatted_list, ite)
            )  # ite,'***',self.func(self.g_best),'***',self.g_best
            inc_counter(new_obf, self.alg, -1, (0, 0, 0), obf_z=obf_z)
            if np.isclose(old_obf, new_obf, rtol=0, atol=10**-3):
                number_equal += 1
            else:
                number_equal = 1
            old_obf = new_obf
            if number_equal == 3:
                break

        #   print(end1-start1,sum_time)
        return self.g_best, self.func(self.g_best)


# Example of the sphere function
# def sphere(x):
#   """
#     In 3D: f(x,y,z) = x² + y² + z²
#   """
#   return np.sum(np.square(x))

# if __name__ == '__main__':
#   init_pos = [1,1,1]
#   PSO_s = PSO(func=sphere, init_pos=init_pos, n_particles=200) #n_particles=50
#   res_s = PSO_s.optimize()
#   print("Sphere function")
#   print(f'x = {res_s[0]}') # x = [-0.00025538 -0.00137996  0.00248555]
#   print(f'f = {res_s[1]}') # f = 8.14748063004205e-06
