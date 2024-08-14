import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
quantize_scope = tfmot.quantization.keras.quantize_scope
LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer


class DefaultLinearQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
        return [(layer.dense.kernel, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        return [(layer.dense.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]


    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in `get_weights_and_quantizers`
        # , in the same order
        layer.dense.kernel = quantize_weights[0]
        # return
    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`
        # , in the same order.
        layer.dense.activation = quantize_activations[0]
        # return
    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}

@tf.keras.saving.register_keras_serializable()
class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32, name="linear", **kwargs):
        """ Initializes the layer """
        super().__init__(name=name, **kwargs)
        self.units = units
        self.dense = tf.keras.layers.Dense(self.units)

    def build(self, input_shape):
        """ Builds the layer """

        # self.dense.build((None, self.units))

    def call(self, inputs):
        """ Calls the layer """
        y = self.dense(inputs)
        return y
class DefaultLinear1QuantizeConfig(DefaultLinearQuantizeConfig):
    pass
@tf.keras.saving.register_keras_serializable()
class Linear1(Linear):
    pass

@tf.keras.saving.register_keras_serializable()
class CustomModel(tf.keras.Model):
    def __init__(self, units, is_quant=False, name="CustomModel", **kwargs):
        """ Initializes the model """
        super(CustomModel, self).__init__(name=name, **kwargs)
        self.units = units
        if is_quant:
            self.layer1 = quantize_annotate_layer(Linear(3, name="Linear_1"), DefaultLinearQuantizeConfig())
            # self.layer1 = Linear(3)
            # self.layer2 = quantize_annotate_layer(Linear1(3, name='Linear_2'), DefaultLinear1QuantizeConfig())
        else:
            self.layer1 = Linear(3)
            # self.layer2 = tf.keras.layers.Dense(3)
        
    def call(self, inputs):
        """ Calls the model """
        y = self.layer1(inputs)
        # y = self.layer2(y)
        return y

    def get_config(self):
        """ Gets the configuration of the model """
        config = super().get_config()
        config.update(
            {
                "units": self.units,
            }
        )
        return config

model = CustomModel(3)
inputs = tf.keras.Input(shape=(3,))
base_model = tf.keras.Model(inputs=inputs, outputs=model.call(inputs))

for v in base_model.trainable_variables:
    v.assign(v*0+1)

model_quant = CustomModel(3, is_quant=True)
base_model_quant = tf.keras.Model(inputs=inputs, outputs=model_quant.call(inputs))


for u,v in zip(base_model.trainable_variables, base_model_quant.trainable_variables):
    v.assign(u)

with quantize_scope(
        {   'DefaultLinearQuantizeConfig': DefaultLinearQuantizeConfig,
            'Linear': Linear},
        {   'DefaultLinear1QuantizeConfig': DefaultLinearQuantizeConfig,
            'Linear': Linear1},
    ):
    quant_aware_model = tfmot.quantization.keras.quantize_apply(base_model_quant)
# quant_aware_model.summary()
# for v in base_model.trainable_variables:
#     print(v)

for v in quant_aware_model.variables:
    print(v.name)
    print(v)
    print("")
# quant_aware_model.variables[0].assign(-1)
# quant_aware_model.variables[1].assign(1)
# W= [[1,1,1],
#     [1,1,1],
#     [1,1,1]]
# b= [[1],
#     [1],
#     [1]]
# x= [[1],
#     [2],
#     [3]]
# y = W*x + b

x = np.array([[1,2,3]], dtype=np.float32)
out = base_model(x)
print(out)
out = quant_aware_model(x)
print(out)
import pdb; pdb.set_trace()