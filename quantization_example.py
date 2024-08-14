import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
quantize_scope = tfmot.quantization.keras.quantize_scope
LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer

# config for quantizing the model
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

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`
        # , in the same order.
        layer.dense.activation = quantize_activations[0]

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}

# Define a custom layer
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

# Define a custom model
@tf.keras.saving.register_keras_serializable()
class CustomModel(tf.keras.Model):
    def __init__(self, units, is_quant=False, name="CustomModel", **kwargs):
        """ Initializes the model """
        super(CustomModel, self).__init__(name=name, **kwargs)
        self.units = units
        if is_quant:
            #quantize the layer
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

if __name__ == "__main__":
    # define the base model
    model = CustomModel(3)
    inputs = tf.keras.Input(shape=(3,))
    base_model = tf.keras.Model(inputs=inputs, outputs=model.call(inputs))

    # force the weights to be ones
    for v in base_model.trainable_variables:
        v.assign(v*0+1)

    # define the quantized model
    model_quant = CustomModel(3, is_quant=True)
    base_model_quant = tf.keras.Model(inputs=inputs, outputs=model_quant.call(inputs))

    # copy the weights from the base model to the quantized model
    for u,v in zip(base_model.trainable_variables, base_model_quant.trainable_variables):
        v.assign(u)

    # apply quantization
    with quantize_scope(
            {   'DefaultLinearQuantizeConfig': DefaultLinearQuantizeConfig,
                'Linear': Linear},
            {   'DefaultLinear1QuantizeConfig': DefaultLinearQuantizeConfig,
                'Linear': Linear1},
        ):
        quant_aware_model = tfmot.quantization.keras.quantize_apply(base_model_quant)
    quant_aware_model.summary()

    # print the variables of the quantized model
    for v in quant_aware_model.variables:
        print(v.name)
        print(v)
        print("")

    # set the input min and max for the quantized model

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

    # test the base model
    x = np.array([[1,2,3]], dtype=np.float32)
    out = base_model(x)
    print("Base model output:")
    print(out)

    # test the quantized model before setting the input min and max
    out = quant_aware_model(x)
    print("Quantized model output:")
    print(out)
    print("The output is wrong because the inputs are clipped [0,0,0]")
    print("if the input min and max are set to [0,0]")

    quant_aware_model.variables[0].assign(-1)
    quant_aware_model.variables[1].assign(1)

    # test the quantized model after setting the input min and max
    out = quant_aware_model(x)
    print("Quantized model output:")
    print(out)
    print("The output is wrong because the inputs are clipped to [1,1,1]")
    print("if the input min and max are set to [-1,1]")
