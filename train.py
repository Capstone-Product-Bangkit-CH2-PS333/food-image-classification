import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import os
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.python.lib.io import file_io
from pathlib import Path
from PIL import ImageFile

print(tf.__version__)
print(tf.test.gpu_device_name())

class MyCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if (logs.get('accuracy')> 0.95 and logs.get('val_accuracy')> 0.95):
        print("\n\nAccuracy reached, stopping the training\n")
        self.model.stop_training = True

class ModelCheckpointInGcs(tf.keras.callbacks.ModelCheckpoint):
    def __init__(
        self,
        filepath,
        gcs_dir: str,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        options=None,
        **kwargs,
    ):
        super().__init__(
            filepath,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            mode=mode,
            save_freq=save_freq,
            options=options,
            **kwargs,
        )
        self._gcs_dir = gcs_dir

    def _save_model(self, epoch, logs):
        super()._save_model(epoch, logs)
        filepath = self._get_file_path(epoch, logs)
        if os.path.isfile(filepath):
            with file_io.FileIO(filepath, mode="rb") as inp:
                with file_io.FileIO(
                    os.path.join(self._gcs_dir, filepath), mode="wb+"
                ) as out:
                    out.write(inp.read())

class FoodImageClassifier():
    def __init__(self, model_file_path = None, train_log_path = None, train_dir = None, testing_dir = None, total_epoch = None, batch_size = None):
        self.model_file_path = model_file_path
        self.train_log_path = train_log_path
        self.train_dir = train_dir
        self.testing_dir = testing_dir
        self.total_epoch = total_epoch
        self.batch_size = batch_size
        self.storage_client = None
        #os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "backend-capstone-406208-a19d3d6fb2b0.json"

    # def download_dataset(self):
    #     print("Downloading datasets\n")

    #     bucket_name = 'image_dataset_backend_foodiefusion'
    #     prefix = 'dataset/'

    #     self.storage_client = storage.Client.from_service_account_json("trainer/backend-capstone-406208-a19d3d6fb2b0.json")
    #     bucket = self.storage_client.get_bucket(bucket_name)
    #     blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
    #     for blob in blobs:
    #         if blob.name.endswith("/"):
    #             continue
    #         file_split = blob.name.split("/")
    #         directory = "/".join(file_split[0:-1])
    #         Path(directory).mkdir(parents=True, exist_ok=True)
    #         blob.download_to_filename(blob.name) 
        
    #     print("Datasets downloaded\n")

    def get_num_samples(self):
        train_foods_dir = os.listdir(self.train_dir)
        print(train_foods_dir)
        print("Total class:", len(train_foods_dir))

        for idx, dir in enumerate(train_foods_dir):
            train_foods_dir[idx] = os.path.join(self.train_dir, dir)

        num_train_samples = 0
        for dir in train_foods_dir:
            num_train_samples += len(os.listdir(dir))
            print(f"There are {len(os.listdir(dir))} images of {dir} for training")
        print()

        testing_food_dir = os.listdir(self.testing_dir)
        for idx, dir in enumerate(testing_food_dir):
            testing_food_dir[idx] = os.path.join(self.testing_dir, dir)

        num_validation_samples = 0
        for dir in testing_food_dir:
            num_validation_samples += len(os.listdir(dir))
            print(f"There are {len(os.listdir(dir))} images of {dir} for testing")
        print()

        return num_train_samples, num_validation_samples

    def create_model(
        self,
        num_train_samples: int,
        num_validation_samples: int,
        ):
        K.clear_session()

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        train_datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(150, 150),
            batch_size=self.batch_size,
            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            self.testing_dir,
            target_size=(150, 150),
            batch_size=self.batch_size,
            class_mode='categorical')


        inception = InceptionV3(weights='imagenet', include_top=False)
        x = inception.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024,activation='relu')(x)
        x = Dense(512,activation='relu')(x)
        x = Dense(128,activation='relu')(x)
        x = Dense(64,activation='relu')(x)
        x = Dropout(0.2)(x)

        predictions = Dense(len(os.listdir(self.train_dir)),kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)

        model = Model(inputs=inception.input, outputs=predictions)
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        # checkpointer = ModelCheckpointInGcs(
        #     filepath=self.model_file_path, 
        #     gcs_dir="gs://food_classification_model_foodiefusion/",
        #     verbose=1, save_best_only=True)
        csv_logger = CSVLogger(self.train_log_path)
        checkpointer = ModelCheckpoint(filepath='best_model_3class.hdf5', verbose=1, save_best_only=True)

        history = model.fit(train_generator,
                            steps_per_epoch = num_train_samples // self.batch_size,
                            validation_data=validation_generator,
                            validation_steps= num_validation_samples // self.batch_size,
                            epochs=self.total_epoch,
                            verbose=1,
                            callbacks=[csv_logger, checkpointer, MyCallback()])

        try:
            model.save(self.model_file_path)
            #self.copy_file_to_bucket(self.model_file_path)
        except:
            pass
        finally:
            return history

    def train_model(self):
        num_train_samples, num_validation_samples = self.get_num_samples()
        history = self.create_model(num_train_samples, num_validation_samples)

    def copy_file_to_bucket(self, file):
        with file_io.FileIO(file, mode='r') as input_f:
            with file_io.FileIO(os.path.join("gs://food_classification_model_foodiefusion", file), mode='w+') as output_f:
                output_f.write(input_f.read())
                print("Saved model.h5 to GCS")

if __name__ == '__main__':
    model = FoodImageClassifier(
        r"model.hdf5",
        r"history.log",
        r"dataset/training/",
        r"dataset/testing/",
        5,
        32,
        )
    
    model.train_model()