from base.base_train import BaseTrain
import os
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, TensorBoard


class SimpleConvModelTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(SimpleConvModelTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_saver()

    def init_saver(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp_name),
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                save_weights_only=True,
                verbose=True,
            )
        )
    
    def train(self):
        history = self.model.fit(
            self.data[0], self.data[1],
            epochs=self.config.num_epochs,
            verbose=True,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            callbacks=self.callbacks,
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])

    def visualize(self):   
        x = range(1, len(self.acc) + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, self.acc, 'b', label='Training acc')
        plt.plot(x, self.val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, self.loss, 'b', label='Training loss')
        plt.plot(x, self.val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig('result.png')
