import unittest

import numpy as np

import vhf
from vhf.lienard_wiechert import LIGHT_SPEED


class TestLW(unittest.TestCase):
    CALCULATOR = vhf.EMFieldCalculator()
    observed_point = np.array([1, 0, 0])
    particle_position = np.ones((10, 3))
    velocity = np.ones((10, 3))
    acceleration = np.ones((10, 3))

    def test_emfield1(self):
        electric_field = self.CALCULATOR.get_electric_field(
            self.observed_point,
            self.particle_position,
            self.velocity,
            self.acceleration
        )
        magnetic_field = self.CALCULATOR.get_magnetic_field(
            self.observed_point,
            self.particle_position,
            self.velocity,
            self.acceleration
        )

        direction, _ = self.CALCULATOR.get_direction_and_distance(
            self.observed_point, self.particle_position)

        a = self.CALCULATOR.cross(direction, electric_field)/ LIGHT_SPEED
        self.assertTrue(np.all(np.allclose(a, magnetic_field)))

    def test_emfield2(self):
        electric_field, magnetic_field = self.CALCULATOR.get_em_field(
            self.observed_point,
            self.particle_position,
            self.velocity,
            self.acceleration
        )
        direction, _ = self.CALCULATOR.get_direction_and_distance(
            self.observed_point, self.particle_position)

        a = self.CALCULATOR.cross(direction, electric_field)/ LIGHT_SPEED
        self.assertTrue(np.all(np.allclose(a, magnetic_field)))