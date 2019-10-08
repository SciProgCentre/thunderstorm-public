import numpy as np

LIGHT_SPEED = 0.3  # m/ns
ELECTRON_CHARGE = 1.6e-19  # SI
EPSILON0 = 1
COULOMB_CONST = 9e+9  # SI


class EMFieldCalculator:

    def __init__(self):
        self.norm = lambda x: np.linalg.norm(x, axis=1)
        self.cross = lambda x, y: np.cross(x, y, axisa=1, axisb=1)
        self.dot = lambda x, y: np.array([np.dot(i, j) for i, j in zip(x, y)])

    def get_direction_and_distance(self, observed_point, particle_position):
        distanse_to_observed_point = self.norm(observed_point - particle_position) # m
        direction_to_observed_point = ((observed_point - particle_position).T / distanse_to_observed_point).T # NO UNIT
        return direction_to_observed_point, distanse_to_observed_point

    def get_electric_field(self, observed_point, particle_position, velocity, acceleration):
        distanse_to_observed_point = self.norm(observed_point - particle_position) # m
        direction_to_observed_point = ((observed_point - particle_position).T / distanse_to_observed_point).T # NO UNIT
        beta = self.norm(velocity) / LIGHT_SPEED  # NO UNIT
        beta_vec_velocity = velocity / LIGHT_SPEED  # NO UNIT
        beta_vec_acc = acceleration / LIGHT_SPEED  # 1/ns
        temp1 = 1 - beta ** 2  # NO UNIT
        temp1 = ((direction_to_observed_point - beta_vec_velocity).T * temp1.T).T  # NO UNIT

        temp2 = distanse_to_observed_point * \
                ((1 - self.dot(direction_to_observed_point, beta_vec_velocity))) ** 3  # m

        firstTerm = (temp1.T / (distanse_to_observed_point * temp2).T).T  # 1/m^2

        temp3 = self.cross(direction_to_observed_point,
                           self.cross(
                               direction_to_observed_point - \
                               beta_vec_velocity, beta_vec_acc))  # 1/ns
        secondTerm = (temp3.T / temp2.T).T / LIGHT_SPEED # 1/m^2
        return (ELECTRON_CHARGE * COULOMB_CONST) * (firstTerm + secondTerm)

    def get_magnetic_field(self, observed_point, particle_position, velocity, acceleration):
        distanse_to_observed_point = self.norm(observed_point - particle_position) # m
        direction_to_observed_point = ((observed_point - particle_position).T / distanse_to_observed_point).T # NO UNIT
        beta = self.norm(velocity) / LIGHT_SPEED  # NO UNIT
        beta_vec_velocity = velocity / LIGHT_SPEED  # NO UNIT
        beta_vec_acc = acceleration / LIGHT_SPEED  # 1/ns

        temp1 = 1 - beta ** 2  # NO UNIT
        temp1 = LIGHT_SPEED*(self.cross(beta_vec_velocity,direction_to_observed_point ).T * temp1.T).T  # NO UNIT

        temp2 = distanse_to_observed_point * \
                ((1 - self.dot(direction_to_observed_point, beta_vec_velocity))) ** 3  # m

        firstTerm = (temp1.T / (distanse_to_observed_point * temp2).T).T  # 1/m^2

        temp3 = self.cross(direction_to_observed_point,
                           self.cross(
                               direction_to_observed_point - \
                               beta_vec_velocity, beta_vec_acc))  # 1/ns
        temp3 = self.cross(direction_to_observed_point, temp3)
        secondTerm = (temp3.T / temp2.T).T # 1/m^2

        return (ELECTRON_CHARGE*1)*(firstTerm + secondTerm)


    def get_em_field(self, observed_point, particle_position, velocity, acceleration):
        electric_field = self.get_electric_field(observed_point, particle_position, velocity, acceleration)
        direction_to_observed_point, _ = self.get_direction_and_distance(observed_point, particle_position)
        magnetic_field = self.cross(direction_to_observed_point, electric_field)/LIGHT_SPEED
        return electric_field, magnetic_field

    def get_retarded_time(self, observed_point, particle_position):
        return self.norm(observed_point-particle_position)/LIGHT_SPEED