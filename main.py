from prettytable import PrettyTable

import numpy as np
from sympy import  sin

def test_func(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2 + 90 * (x[3] - x[2]) ** 2 + (1 - x[2]) ** 2 + 10.1 * (
            (x[1] - 1) ** 2 + (x[3] - 1) ** 2) + 19.8 * (x[1] - 1) * (x[3] - 1)

#глобальный мин - f(0;0)=0.0
def func_1_perfectly(x):
    return x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2



# точное решение  f(1,1)=0
def func_2_acceptable(x):
    return (9*x[0]**2+2*x[1]**2-11)**2+(3*x[0]+4*x[1]**2-7)**2




#глобальный мин - f(-0.54719;-1.54719)=-1.9133
def func_3_bad(x):
    return sin(x[0]+x[1])+(x[0]-x[1])**2-1.5*x[0]+2.5*x[1]+1
    #return -(x[1]+47)*sin((abs(x[0]/2+(x[1]+47)))**0.5)-x[0]*sin((abs(x[0]-(x[1]+47)))**0.5)


def harmony(func, hms, hmcr, par_initial, fw, count, n):
    """Алгоритм Harmony Search"""

    # Шаг 1. Инициализация
    k = 0

    # Шаг 2. Формирование начального множества решений
    # Генерируем hms случайных решений в допустимом пространстве
    hm = np.random.uniform(np.min(fw), np.max(fw), (hms, n))

    # Вычисляем значения целевой функции для каждого решения
    func_value = np.apply_along_axis(func, 1, hm)

    # Добавляем значения целевой функции к решениям
    hm = np.hstack((hm, func_value.reshape(-1, 1)))

    # Возвращает индексы максимальных значений среди func_value
    x_worst = np.argmax(func_value)

    c = np.log(np.abs(np.min(fw) / np.max(fw))) / count

    while True:

        # Линейное увеличение par от начального значения до 1.0
        par = par_initial + (1.0 - par_initial) * (k / count)

        fw = [np.max(fw) * np.exp(c * k)] * n

        # Шаг 3. Генерация нового решения
        x_prime = np.zeros(n)
        x_new = np.zeros(n)

        # Шаг 3.1. Получить значения x_i
        for i in range(n):

            # Обновление значения hmcr
            hmcr = hmcr * np.exp(c * k)

            # Генерация случайного числа для выбора
            u = np.random.uniform(0, 1)
            if u < hmcr:  # С вероятностью hmcr выбираем значение из памяти гармонии
                # Выбираем индекс из памяти гармонии
                idx = int(u * hms)
                x_prime[i] = hm[idx, i]

                # Шаг 3.2. Если значение выбрано из памяти
                u_offset = np.random.uniform(-1, 1)  # Генерируем случайное число из [-1, 1]
                u_par = np.random.uniform(0, 1)  # Для проверки, что с вероятностью par

                if u < par:  # С вероятностью par обновляем значение

                    x_new[i] = x_prime[i] + fw[i] * u_offset
                else:  # С вероятностью 1 - par
                    x_new[i] = x_prime[i]



            else:  # С вероятностью 1 - hmcr выполнить

                x_prime[i] = np.random.uniform(np.min(fw), np.max(fw))  # Выбираем значение внутри промежутка [a_i, b_i]
                x_new[i] = x_prime[i]  # Если не выбирали из памяти, просто копируем значение

        # Шаг 4. Обновление памяти гармонии
        # Вычисляем значение целевой функции для нового решения
        new_func_value = func(x_new)

        if new_func_value < hm[x_worst, -1]:  # Если новое решение лучше наихудшего
            hm[x_worst, -1] = new_func_value  # Обновляем значение функции
            hm[x_worst, : -1] = x_new  # Обновляем решение (без значения функции)

        # Шаг 5. Проверка условия окончания
        if k == count - 1:
            break

        k += 1
    best_x_idx = np.argmin(hm[:, -1])

    return hm[best_x_idx, : -1], hm[best_x_idx, -1]


def pretty_table(iterable):
    table = PrettyTable()
    table.field_names = ["k", "x", "f(x)"]
    for k, x, f in iterable:
        table.add_row([k, x, f])
    return table


def test():
    func = {
        (test_func, 4): (20, 0.95, 0.7, [1e-10, 2], 10000, 4),
        (func_1_perfectly, 5): ( 50, 0.95, 0.35, [1e-5, 50], 10000, 5),
        (func_2_acceptable, 2): (40, 0.95, 0.85, [1e-30, 4], 10000, 2),
        (func_3_bad, 2): (30, 0.85, 0.7, [1e-30, 5.5], 100000, 2)
    }
    # тоже самое внутри func сделать для func_1_perfectly, func_2_acceptable и func_3_bad

    for (func_name, size), (hms, hmcr, par_initial, bounds, count, n) in func.items():
        a = []
        iterable_data = []
        for i in range(10):

            x_best, min_value_func = harmony(func=func_name, hms=hms, hmcr=hmcr, par_initial=par_initial,
                                             fw=np.random.uniform(bounds[0], bounds[1], size=n), count=count, n=n)
            iterable_data.append((i + 1, x_best, min_value_func))
            a.append(x_best)

        mean_a = np.mean(a, axis=0)

        min_value_func = func_name(mean_a)
        print(pretty_table(iterable_data))
        print(f"Оптимальное решение для функции {func_name.__name__}: {mean_a} значение функции: {min_value_func}")


if __name__ == "__main__":
    test()