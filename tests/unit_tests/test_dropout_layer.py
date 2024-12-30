import numpy as np
from unittest import TestCase
from si.neural_networks.layers import Dropout


class TestDropoutLayer(TestCase):

    def setUp(self):
        np.random.seed(10)
        self.probability = 0.5
        self.input_data = np.random.rand(100, 100)
        self.dropout_layer = Dropout(self.probability)

    def test_training_mode(self):
        output = self.dropout_layer.forward_propagation(self.input_data, training=True)

        self.assertTrue(np.any(output == 0), "None of the neurons were deactivated in training mode")

        self.assertTrue(np.all((output == 0) | (output == self.input_data * self.dropout_layer.mask)),
                        "The active neurons dont match the input in training mode")

        self.assertEqual(output.shape, self.input_data.shape, "The output shape doesn't match the input shape")

    def test_inference_mode(self):
        output = self.dropout_layer.forward_propagation(self.input_data, training=False)

        np.testing.assert_array_almost_equal(output, self.input_data,
                                             err_msg="The output in inference mode has been modified")

        self.assertEqual(output.shape, self.input_data.shape, "The output shape doesn't match the input shape")

    def test_mask_generation(self):
        self.dropout_layer.forward_propagation(self.input_data, training=True)
        mask = self.dropout_layer.mask

        self.assertTrue(np.all((mask == 0) | (mask == 1)), "The mask contains values other than 0 or 1")

        active_rate = np.mean(mask)
        expected_active_rate = 1 - self.probability
        self.assertAlmostEqual(active_rate, expected_active_rate, delta=0.1,
                               msg=f"The activation rate ({active_rate:.2f}) doesnt match the expected "
                                   f"({expected_active_rate:.2f})")

        self.assertEqual(mask.shape, self.input_data.shape,
                         "The shape of the mask doesn't match the shape of the input")
