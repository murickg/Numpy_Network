import time
import numpy as np
from nn.loss_functions.hinge_loss import hinge_loss
from optimization.adam_optimize import Adam


def progress_bar(iterable, text='Epoch progress', end=''):
    """Мониториг выполнения эпохи

    ---------
    Параметры
    ---------
    iterable
        Что-то по чему можно итерироваться

    text: str (default='Epoch progress')
        Текст, выводящийся в начале

    end : str (default='')
        Что вывести в конце выполнения
    """
    max_num = len(iterable)
    iterable = iter(iterable)

    start_time = time.time()
    cur_time = 0
    approx_time = 0

    print('\r', end='')

    it = 0
    while it < max_num:
        it += 1
        print(f"{text}: [", end='')

        progress = int((it - 1) / max_num * 50)
        print('=' * progress, end='')
        if progress != 50:
            print('>', end='')
            print(' ' * (50 - progress - 1), end='')
        print('] ', end='')

        print(f'{it - 1}/{max_num}', end='')
        print(' ', end='')

        print(f'{cur_time}s>{approx_time}s', end='')

        yield next(iterable)

        print('\r', end='')
        print(' ' * (60 + len(text) + len(str(max_num)) + len(str(it)) \
                     + len(str(cur_time)) + len(str(approx_time))),
              end='')
        print('\r', end='')

        cur_time = time.time() - start_time

        approx_time = int(cur_time / it * (max_num - it))
        cur_time = int(cur_time)
        print(end, end='')


# TODO: изменить под свой фреймворк
def gradient_check(X, y, model, epsilon=1e-7, seed=42):
    np.random.seed(seed)
    optimizer = Adam(model.parameters())
    for param in model.parameters():
        loss_func = hinge_loss(model(X), y)
        optimizer.zero_grad()
        loss_func.backward()
        backprop_grad = param.grads.flatten()

        central_difference = []
        for idx in progress_bar(range(param.params.size)):
            cur_idx = np.unravel_index(idx, param.params.shape)
            cur_params = np.copy(param.params[cur_idx])
            pos_params = cur_params + epsilon
            neg_params = cur_params - epsilon

            param.params[cur_idx] = pos_params
            np.random.seed(seed)
            pos_loss_func = hinge_loss(model(X), y).loss
            param.params[cur_idx] = neg_params
            np.random.seed(seed)
            neg_loss_func = hinge_loss(model(X), y).loss
            param.params[cur_idx] = cur_params

            central_difference.append((pos_loss_func - neg_loss_func) / (2 * epsilon))

    central_difference = np.array(central_difference)

    numerator = np.linalg.norm(central_difference - backprop_grad)
    denominator = np.linalg.norm(central_difference) + np.linalg.norm(backprop_grad) + epsilon
    diff = numerator / denominator
    if diff > epsilon:
        print('Backprop is incorrect!')
    else:
        print('Backprop is correct!')
    np.random.seed(seed)
    return diff
