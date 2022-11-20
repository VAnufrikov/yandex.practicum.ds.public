from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_train(path):  # Загружаем данные
    train_datagen = ImageDataGenerator(validation_split=0.25,
                                       rescale=1. / 255,  # Добавим Аугментацию(увеличения выборки) улучшаем качество
                                       horizontal_flip=True,  # Отражаем по горизонтали
                                       vertical_flip=True)  # Отражаем по вертикали

    train_datagen_flow = train_datagen.flow_from_directory(  # Извлекаем данные из папки
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        seed=12345)

    return train_datagen_flow


def create_model(input_shape):  # Создаем модель по слоям
    model = Sequential()

    model.add(Conv2D(filters=6,  # Количество фильтров, которому равна величина выходного тензора
                     kernel_size=(3, 3),  # Пространственный размер фильтра K. Фильтр — это тензор размером K×K×D,
                     # где D равна глубине входного изображения.
                     activation='relu',  # Активация, которую применяют сразу же после свёртки с фильтром
                     input_shape=(150, 150, 3))
              )  # Создаем сверточный слой Conv2D для изображения 150x150x3

    model.add(AvgPool2D(pool_size=(2, 2)))  # Выполняем усреднение

    model.add(Conv2D(filters=16,
                     kernel_size=(3, 3),
                     activation='relu')
              )

    model.add(AvgPool2D(pool_size=(2, 2)))  # Выполняем усреднение

    model.add(Flatten())  # Разглаживаем многомерный тензор и делаем одномерным(нужно для передачи полносвязному слою)

    model.add(Dense(units=24,
                    activation='relu')
              )  # Полносвязный слой с активацией relu

    model.add(Dense(units=18,
                    activation='relu')
              )  # Полносвязный слой с активацией relu

    model.add(Dense(units=12,
                    activation='softmax')
              )  # Полносвязный слой с активацией softmax

    optimizer = Adam(lr=0.001)  # Заменяем SGD на алгоритм ADAM так как он более оптимален для этой задачи

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc']
                  )

    return model


def train_model(model,
                train_datagen_flow,
                test_data,
                batch_size=None,
                epochs=45,
                steps_per_epoch=None,
                validation_steps=None):  # Обучаем созданную модель
    if steps_per_epoch is None:
        steps_per_epoch = len(train_datagen_flow)
    if validation_steps is None:
        validation_steps = len(test_data)

    model.fit(train_datagen_flow,
              validation_data=test_data,
              batch_size=batch_size,  # Размер батча
              epochs=epochs,  # Количество эпох
              steps_per_epoch=steps_per_epoch, # Количество шагов обучения для каждой эпохи
              validation_steps=validation_steps,
              verbose=2,
              shuffle=True) # Перемешивание

    return model
