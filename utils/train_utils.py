import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.image import random_crop, resize_with_crop_or_pad
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy

import time

class timer(tf.keras.callbacks.Callback):
    '''
        Simle call back class to track total training time.
    '''
    def __init__(self):
        super().__init__()
        
        self.start_time = time.perf_counter()
    
    def on_epoch_end(self, epoch, logs=None):
        ''' Help keep track of total training time needed for various models. '''   
        if epoch % 25 == 0:
            end_time = time.perf_counter()
            run_time = end_time - self.start_time
            hrs, mnts, secs = int(run_time // 60 // 60), int(run_time // 60 % 60), int(run_time % 60)

            template = 'Epoch: {:04}, Total Run Time: {:02}:{:02}:{:02}'
            template += ' - Loss: {:.4e}, Accuracy: {:.3f}, Test Loss: {:.4e}, Test Accuracy: {:.3f}'

            train_loss, train_accuracy = logs['loss'], logs['accuracy']
            test_loss, test_accuracy = logs['val_loss'], logs['val_accuracy']
            print(template.format(epoch, hrs, mnts, secs, train_loss, train_accuracy, test_loss, test_accuracy))
        
class inverse_squareroot_lr:
    ''' 
        This is the learning rate used with SGD in the paper (Inverse square root decay). 
        Learning Rate starts at 0.1 and then drops every 512 batches.
    '''
    def __init__(self, n_steps=512, init_lr=0.1):
        self.n = n_steps
        self.gradient_steps = 0
        self.init_lr = init_lr
        
    def __call__(self):
        lr = self.init_lr / tf.math.sqrt(1.0 + tf.math.floor(self.gradient_steps / self.n))
        self.gradient_steps += 1
        return lr
    


class Model_Trainer:
    # Training Wrapper For Tensorflow Models. Allows a predifined model to be easily trained
    # while also tracking parameter and gradient information.
    
    # Please ensure that model_id is unique. It provides the path for all model statistics.
    
    """
        NO longer in use, model.fit provides significant training speed up. the features utilized below 
        will be replaced with tensorflow callbacks.
    """
    
    def __init__(self, model, model_id, lr=1e-4, optimizer=None, data_augmentation=None):
        '''
            Parameters
            ----------
            
            model: tensorflow.keras.Model
            model_id : string
                An identifying string used in saving model metrics. 
            lr : float, tensorflow.keras.optimizers.schedules
                If using the default optimizer, this is the lr used in the Adam optimizer. 
                This value is ignored if an optimizer is passed to the trainer.
            optimizer : tensorflow.keras.optimizers
                A pre-defined optimizer used in training the neural network
            data_augmentation : tensorflow.keras.Sequential
                A tensorflow model used to perform data augmentation during training. 
                See here: https://www.tensorflow.org/tutorials/images/data_augmentation#use_keras_preprocessing_layers                
        '''
                
        self.lr = lr   
        
        self.model = model
        self.init_loss()

        # Can optionally pass a seperate optimizer.
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.init_optimizer()
            
        if data_augmentation is not None:
            self.is_data_augmentation = True
            self.data_augmentation = data_augmentation
        else:
            self.is_data_augmentation = False
            self.data_augmentation = None
            
        # Used to save the parameters of the model at a given point of time.
        self.checkpoint = tf.train.Checkpoint(model=self.model)
        self.checkpoint_path = self.model.__class__.__name__ + '/' + model_id + "/training_checkpoints"


        self.summary_path = self.model.__class__.__name__ + '/' + model_id + '/summaries/'
        self.summary_writer = tf.summary.create_file_writer(self.summary_path)
        
        self.gradients = None
        
    
    #initialize loss function and metrics to track over training
    def init_loss(self):
        self.loss_function = SparseCategoricalCrossentropy()

        self.train_loss = Mean(name='train_loss')
        self.train_accuracy = SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = Mean(name='test_loss')
        self.test_accuracy = SparseCategoricalAccuracy(name='test_accuracy')
        

    # Initialize Model optimizer
    def init_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, epsilon=1e-8)
    
    
    # Take a single Training step on the given batch of training data.
    def train_step(self, images, labels, track_gradient=False):
     
        with tf.GradientTape() as gtape:
            predictions = self.model(images, training=True)
            loss = self.loss_function(labels, predictions)
            
        gradients = gtape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Track model Performance
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)
        
        return self.train_loss.result(), self.train_accuracy.result()*100
    
    # Evaluate Model on Test Data
    def test_step(self, data_set):
        predictions = self.model.predict(images)
        test_loss = self.loss_function(labels, predictions)
        
        self.test_loss(test_loss)
        self.test_accuracy(labels, predictions) 
        
        return self.test_loss.result(), self.test_accuracy.result()*100
        
    # Reset Metrics 
    def reset(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        
    # Save a checkpoint instance of the model for later use
    def model_checkpoint(self):
        # Save a checkpoint to /tmp/training_checkpoints-{save_counter}
        save_path = self.checkpoint.save(self.checkpoint_path)
        return save_path

    def log_metrics(self, ):
        # Log metrics using tensorflow summary writer. Can Then visualize using TensorBoard
        step = self.checkpoint.save_counter
        
        with self.summary_writer.as_default():
            tf.summary.scalar('Train Loss', self.train_loss.result(), step=step)
            tf.summary.scalar('Train Accuracy', self.train_accuracy.result(), step=step)
            tf.summary.scalar('Test Loss', self.test_loss.result(), step=step)
            tf.summary.scalar('Test Accuracy', self.test_accuracy.result(), step=step)
                        
            
def load_data(data_set, batch_size, label_noise, augment_data=True):
    '''
        Helper Function to Load data in the form of a tensorflow data set, apply label noise, and return the 
        train data, test data, and the dataset info objet.
                
        Available datasets can be found here: https://www.tensorflow.org/datasets
    '''
    
    # Load the Data Set
    (ds_train, ds_test), ds_info = tfds.load(
        data_set,
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    
    n_classes = ds_info.features['label'].num_classes
    N = ds_info.splits['train'].num_examples
    
    if augment_data:
        ds_train = augment_data_set(ds_train)
    
    def add_label_noise(image, label):
        # helper function to add label noise and cast data to correct types.
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.float32)
        if label_noise < tf.random.uniform([], 0, 1):
            return (image, label)  
        else:
            return (image, tf.random.uniform(shape=(), minval=0, maxval=n_classes, dtype=tf.float32))  
    
    # Shuffle the Data set and set training batch size 
    ds_train = ds_train.map(add_label_noise).batch(batch_size)
    
    # Initilialize the Test Training data set
    ds_test = ds_test.batch(batch_size).map(
        lambda image, label: (tf.cast(image, tf.float32), tf.cast(label, tf.float32))
    ).cache()
    
    return ds_train, ds_test, ds_info    


def augment_data_set(data_set, crop_dim=32, target_height=36, target_width=36):
    ''' Apply random cropping and random horizontal flip data augmentation as done in Deep Double Descent '''
   
    # random flip. Preserve original label
    rand_flip = lambda image, label: (
        tf.image.random_flip_left_right(image), label
    )
    
    # Random Crop. Preserve original label
    def rand_crop(image, label):
        offset_height = tf.random.uniform([], 0, 3, dtype=tf.int32)
        offset_width = tf.random.uniform([], 0, 3, dtype=tf.int32)
        image = tf.image.resize_with_crop_or_pad(image, target_height, target_width)
        return (tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_dim, crop_dim), label)
    
    data_set_flip = data_set.map(rand_flip)
    data_set_drop = data_set.map(rand_crop)
    
    return {
        'flip': data_set_flip,
        'crop': data_set_drop
    }
