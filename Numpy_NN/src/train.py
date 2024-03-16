import numpy as np
from sklearn.metrics import accuracy_score
from training.epoch_information import EpochInformation
import time
from optimization.gd_optimizer import GD
from optimization.adam_optimize import Adam
from dataset.dataloader import Dataloader
from utils import progress_bar
from nn.loss_functions.mse_loss import mse_loss
from nn.loss_functions.hinge_loss import hinge_loss

def train(dataset, model, epochs=100, lr=1e-3, batch_size=1000,
          valid_dataset=None, timer=True, return_history=True, step=1,
          visualize_train=True, optim_method="Adam", alpha1=None, alpha2=None):
    """Тренирует модель

    dataset
        Массив данных для обучения

    model
        Модель, которую будем обучать

    epochs : int (default=100)
        Число эпох

    lr : float (default=1e-4)
        Learning rate

    batch_size : int (default=1000)
        Размер бача
        Если парамтр равен -1, то берутся все данные

    valid_dataset (default=None)
        Массив данных для валидации
        Если установлен в None, оценки на валидационной выборке нет

    timer : bool (default=True)
        Если установлен в True будет выводить сколько времени заняло
        обучение на данной эпохе

    return_history : bool (default=True)
        Если установлен в True, вернет историю изменений ошибки

    step : int (default=1)
        Число итераций, через которое вычисляются ошибки

    visualize_train : bool (default=True)
        Визуализировать ли метрики во время обучения

    optim_method : str (default=Adam)
        Метод оптимизации, доступны
        * "Adam" -- алгоритм Adam
        * "GD" -- градиентный спуск

    alpha1 : float (default=None)
        Если не None, то применяет l_1 регуляризацию
        с параметром alpha_1

    alpha2 : float (default=None)
        Если не None, то применяет l_2 регуляризацию
        с параметром alpha_2
    ----------
    Возвращает
    ----------
    * Если return_history=True и valid_dataset не None
        train_loss_history : list
            История ошибок на обучающей выборке

        valid_loss_history : list
            История ошибок на валидационной выборке

        train_acc_history : list
            История точности предсказания на обучающей выборке

        valid_acc_history : list
            История точности предсказания на валидационной выборке

    * Если return_history=True и valid_dataset=None
        train_loss_history : list
            История ошибок на обучающей выборке

        train_acc_history : list
            История точности предсказания на обучающей выборке

    * Если return_history=False
        None
    """
    if timer:
        start_time = time.time()

    if optim_method == "Adam":
        optimizer = Adam(model.parameters(), lr=lr,
                         alpha1=alpha1, alpha2=alpha2)
    elif optim_method == "GD":
        optimizer = GD(model.parameters(), lr=lr,
                       alpha1=alpha1, alpha2=alpha2)

    if return_history:
        train_loss_history = []
        train_acc_history = []
        if valid_dataset:
            valid_loss_history = []
            valid_acc_history = []

    description = [('Train loss', 0, 4)]
    description.append(('Train acc', 0, 4))

    if valid_dataset:
        description.append(('Valid loss', 0, 4))
        description.append(('Valid acc', 0, 4))

    description.append(('Grad/W', 10, 6))

    if timer:
        description.append(('Time, s', 10, 1))
        description.append(('Total, s', 10, 1))

    if visualize_train:
        epoch_logger = EpochInformation(epochs, description)

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        dataloader = Dataloader(dataset, batch_size=batch_size)
        train_loss = 0
        sample_size_train = 0
        for vecs, labels in progress_bar(dataloader):
            # TODO: Реализовать обучение модели на батче данных
            out = model.forward(vecs)
            Loss = hinge_loss(out, labels)
            train_batch_loss = Loss.loss * len(labels)
            train_loss += train_batch_loss
            sample_size_train += len(labels)
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

        if (epoch + 1) % step == 0 or (epoch + 1) == epochs:
            if (epoch + 1) == epochs and ((epoch + 1) % step != 0):
                step = (epoch + 1) % step
            values = {}

            model.eval()

            dataloader = Dataloader(dataset, batch_size=len(dataset),
                                    is_train=False)

            train_acc = 0
            sample_size_eval = 0
            for vecs, labels in progress_bar(dataloader,
                                             text='Evaluating train'):
                # TODO: Реализовать рассчет accuracy модели на батче данных
                out = model.forward(vecs).array
                pred = np.argmax(out, axis=-1)
                train_acc += np.sum(pred == labels)
                sample_size_eval += len(labels)
            train_loss = train_loss / sample_size_train
            train_acc = train_acc / sample_size_eval
            values['Train loss'] = train_loss
            values['Train acc'] = train_acc

            num_params = 0
            scale = 0

            for param in model.parameters():
                param_norm = np.linalg.norm(param.params.flatten())
                if param_norm == 0.:
                    continue
                else:
                    num_params += 1
                    grad_norm = np.linalg.norm(param.grads.flatten())
                    scale += lr * grad_norm / param_norm

            scale /= num_params

            values['Grad/W'] = scale

            if return_history:
                train_loss_history.append(train_loss)
                train_acc_history.append(train_acc)

            if valid_dataset:

                valid_dataloader = Dataloader(valid_dataset,
                                              batch_size=len(valid_dataset),
                                              is_train=False)
                valid_loss, valid_acc = 0, 0
                sample_size_valid = 0
                for vecs, labels in progress_bar(valid_dataloader,
                                                 text='Evaluating valid'):
                    # TODO: Реализовать рассчет accuracy модели на батче данных
                    out = model.forward(vecs)
                    valid_loss += hinge_loss(out, labels).loss * len(labels)
                    pred = np.argmax(out.array, axis=-1)
                    valid_acc += np.sum(pred == labels)
                    sample_size_valid += len(labels)
                valid_loss = valid_loss / sample_size_valid
                valid_acc = valid_acc / sample_size_valid
                values['Valid loss'] = valid_loss
                values['Valid acc'] = valid_acc

                if return_history:
                    valid_loss_history.append(valid_loss)
                    valid_acc_history.append(valid_acc)

            if timer:
                values['Time, s'] = time.time() - epoch_start_time
                values['Total, s'] = time.time() - start_time

            if visualize_train:
                epoch_logger.update(values, step)

    if return_history:
        if valid_dataset:
            return train_loss_history, valid_loss_history, \
                   train_acc_history, valid_acc_history
        else:
            return train_loss_history, train_acc_history
    else:
        return None