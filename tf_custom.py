"""Tensorflow Customised Functions"""

from copy import deepcopy
import math
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import types


def import_transfer_method(function):
    """Imports an LFMC model class
    

    Parameters
    ----------
    model_class : str
        The name of the LFMC model class.

    Returns
    -------
    class
        The imported LFMC model class.

    """
    module = __import__('tf_custom', fromlist=[function])
    return getattr(module, function)


def sourcerer(source_model, target_size, source_size, target_max=1e6):
    def source_reg_loss(y_true, y_pred):
        loss = loss_fn(y_true, y_pred)
        if isinstance(loss, types.FunctionType):
            loss = tf.reduce_mean(loss)
        sq_diff_reg = 0
        target_weights = source_model.model.trainable_weights
        for i in range(len(source_weights)):
            sw = source_weights[i]
            tw = target_weights[i]
            sq_diff = tf.reduce_sum(tf.math.squared_difference(sw, tw))
            sq_diff = tf.cast(sq_diff, dtype=loss.dtype)
            if scale_variance:
                sq_diff_reg += lambda_ * sq_diff * tf.math.reduce_variance(y_true)
            else:
                sq_diff_reg += lambda_ * sq_diff
        total_loss = loss + tf.cast(sq_diff_reg, dtype=loss.dtype)
        return total_loss

    transfer_params = source_model.params['transferModel']
    freeze_bn = transfer_params.get('freeze_bn', True)
    transfer_limit = transfer_params.get('limit', None)
    scale_variance = transfer_params.get('scale', False)
    loss_fn = tf.keras.losses.get(source_model.params['loss'])
    diagnostics = source_model.params['diagnostics']

    if freeze_bn:  # this freezes the batch normalisation layers
        for layer in source_model.model.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                if diagnostics:
                    print(f'{layer.name} frozen.')
                layer.trainable = False

    if (transfer_limit is not None) and (transfer_limit is not False):  
        # set target_max to source_size scaled by transfer_limit
        target_max = min(target_max, source_size * transfer_limit)
    if target_max > 1:  # apply regularisation if target_max > 1
        lambda_ = 1e10 * np.power(float(target_size),
                                  (math.log(1e-20) /
                                   math.log(target_max)))
        if diagnostics:
            print("Lambda value for regularization: ", lambda_)
        source_weights = deepcopy(source_model.model.trainable_weights)
        source_model.losses = source_reg_loss
    elif diagnostics:
        print(f"Regularizer disabled; limit: {transfer_limit}; target_max: {target_max}")
        
    source_model.compile()
    source_model.losses = None


class AdabnCallback(keras.callbacks.Callback):
    ''' Callback class for AdaBN Domain Adapatation
    
    Resets the BN gamma and beta weights back to the learned weights.
    
    It would be better to freeze these, but there doesn't appear to be
    any way to do this without also freezing the running mean/variance,
    which need to be updated.
    '''
    saved_weights = []
    saved_model = None
    moving_mean = []
    moving_variance = []
    def on_train_batch_end(self, batch, logs=None):
        if self.diagnostics:
            print('on_train_batch_end')
        if (batch + 1) * self.batch_size <= self.target_size:
            this_batch = self.batch_size
        else:
            this_batch = self.target_size - batch * self.batch_size
        
        saved_layer = 0
        sum_x2 = []
        sum_xbar = []
        for layer in self.saved_model.model.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                weights = layer.get_weights()
                new_weights = self.saved_weights[saved_layer].copy() #+ weights[2:]
                shape = weights[0].shape[0]
                for w in ['moving_mean', 'moving_variance']:
                    new_weights.append(getattr(layer, f'{w}_initializer')(shape).numpy())
                layer.set_weights(new_weights)
                means = weights[2]
                variances = weights[3]
                sum_x2.append((variances + np.square(means)) * this_batch)
                sum_xbar.append(means * this_batch)
                saved_layer += 1
                if (saved_layer == 1) and self.diagnostics:
                    print(batch, this_batch, self.batch_size, self.target_size)
                    print(new_weights)
                    print(means)
                    print(variances)
        self.moving_mean.append(sum_xbar)
        self.moving_variance.append(sum_x2)
        
    def on_epoch_end(self, epoch, logs=None):
        # Calculate mm/mv for each layer and update weights
        if self.diagnostics:
            print('on_epoch_end')
        saved_layer = 0
        for layer in self.saved_model.model.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                weights = layer.get_weights()
                sum_x2 = np.zeros(weights[0].shape)
                sum_xbar = np.zeros(weights[0].shape)
                for batch in range(len(self.moving_mean)):
                    sum_xbar += self.moving_mean[batch][saved_layer]
                    sum_x2 += self.moving_variance[batch][saved_layer]
                means = sum_xbar / self.target_size
                variances = sum_x2 / self.target_size - np.square(means)
                new_weights = self.saved_weights[saved_layer] + [means, variances]
                layer.set_weights(new_weights)
                saved_layer += 1
                if (saved_layer == 1) and self.diagnostics:
                    print(new_weights)
                    print(means)
                    print(variances)
        
    def on_epoch_begin(self, epoch, logs=None):
        if self.diagnostics:
            print('on_epoch_begin')
        self.moving_mean = []
        self.moving_variance = []
        saved_layer = 0
        for layer in self.saved_model.model.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                weights = layer.get_weights()
                shape = weights[0].shape[0]
                new_weights = []
                # Restore saved gamma and beta values
                new_weights = self.saved_weights[saved_layer].copy() #+ weights[2:]
                # Re-initialise moving_mean and moving_variance
                for w in ['moving_mean', 'moving_variance']:
                    new_weights.append(getattr(layer, f'{w}_initializer')(shape).numpy())
                # Set the new layer weights
                layer.set_weights(new_weights)
                saved_layer += 1
                if (saved_layer == 1) and self.diagnostics:
                    print(new_weights)

        
def adabn(source_model, target_size, source_size, pre_transfer=False):
    # Note: this should always be run with epochs=1 and stepsPerExec=1
    diagnostics = source_model.params['diagnostics']
    
    # Set callbacks to restore gamma and beta
    AdabnCallback.saved_model = source_model
    AdabnCallback.target_size = target_size
    AdabnCallback.batch_size = source_model.params['batchSize']
    AdabnCallback.moving_mean = []
    AdabnCallback.moving_variance = []
    AdabnCallback.diagnostics = diagnostics
    source_model.callback_list.append(AdabnCallback())
    
    layers_trainable = []
    layers_momentum = []
    for layer in source_model.model.layers:
        layers_trainable.append(layer.trainable)
        if isinstance(layer, keras.layers.BatchNormalization):
            # Save gamma and beta for use in callbacks
            layers_momentum.append(layer.momentum)
            layer.momentum = 0  # Completely overwrite moving mean/var in each batch
            weights = layer.get_weights()
            AdabnCallback.saved_weights.append(weights[:2]) #[:2])
            if diagnostics:
                print(layer._name)
                print(f'Current weights: {weights}')
        else: # Freeze all non-BN layers
            if diagnostics:
                print(f'{layer.name} frozen.')
            layer.trainable = False
            layers_momentum.append(None)
    # Compile the updated model; ensure callbacks are run after each batch
    source_model.compile(steps_per_execution=1)
    
    # If set as a pre-transfer method, train the model for a single epoch to set the
    # moving mean/variance, then remove the AdaBN callback and unfreeze the layers
    if pre_transfer:
        if diagnostics:
            print('Adjusting moving mean and variance')
        source_model.last_epoch = 1  # Just train model for a single epoch
        source_model.train(source_model.train_data['X'], source_model.train_data['y'],
                            source_model.val_data['X'], source_model.val_data['y'])
        source_model.callback_list = source_model.callback_list[:-1]
        for num, layer in enumerate(source_model.model.layers):
            layer.trainable = layers_trainable[num]
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.momentum = layers_momentum[num]
        source_model.compile()


def freeze_layers(source_model, target_size, source_size):
    transfer_params = source_model.params['transferModel']
    diagnostics = source_model.params['diagnostics']
    freeze = transfer_params.get('layers', [])
    if diagnostics:
        print(f'Freezing layers {freeze} - model has {len(source_model.model.layers)} layers.')
    for layer in source_model.model.layers:
        if any([substr_ in layer.name for substr_ in freeze]):
            if diagnostics:
                print(f'{layer.name} frozen.')
            layer.trainable = False
        elif diagnostics:
            if layer.trainable:
                print(f'{layer.name} trainable.')
            else:
                print(f'{layer.name} not trained.')
    source_model.compile()

