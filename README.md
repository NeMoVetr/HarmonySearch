# Название метода: Гармонический поиск (Harmony Search)

## Историческая справка

В 2001 году Geem разработал и предложил алгоритм гармонического поиска (Harmony Search, или HS). Некоторые авторы утверждают, что алгоритм был вдохновлен игрой джаз-музыкантов, другие полагают, что в основе лежит процесс создания приятной мелодии. Это и легло в основу алгоритма.

## Возможности метода

Находит как глобальный, так и локальный минимум в зависимости от начальных данных:
- **Локальный:** Если начальная точка или гармония находится близко к локальному минимуму, алгоритм может застрять в нем, особенно если параметры (например, коэффициент гармонического поиска или HMCR) настроены так, что процесс склоняется к более детализированным локальным решениям.
- **Глобальный:** Метод способен избежать застревания в локальных минимумах, особенно если используются случайные элементы в процессе поиска (параметр PAR), которые настроены должным образом.

## Типы функций

Подходит для работы с:
- Многомерными функциями.
- Негладкими функциями.
- Функциями непрерывной и дискретной оптимизации.

## Плюсы метода

- Простота реализации и понимания.
- Малое количество настраиваемых параметров и рекомендации по их выбору.
- Метод нулевого порядка (нет необходимости в численном дифференцировании).
- Удобство встраивания в другие методы (например, меметические алгоритмы или генетические алгоритмы).

## Минусы метода

- Низкая скорость сходимости.
- Плохо применим для задач больших размерностей из-за простоты составляющих.

## Алгоритм метода

### Шаг 1. Инициализация алгоритма
Задать параметры метода:
- `hms` — размер памяти гармонии;
- `hmcr` — частота выбора значений из памяти гармонии;
- `par` — частота выбора соседнего значения;
- `fw` — вектор максимального изменения приращения;
- `K` — максимальное число итераций.

Установить начальное число итераций `k = 0`.

### Шаг 2. Формирование начального множества решений
На множестве допустимых решений сгенерировать `hms` решений `\( x_1, \ldots, x_{hms} \)`. 
Вычислить соответствующие значения целевой функции `\( f(x_1), \ldots, f(x_{hms}) \)` и сохранить их в памяти гармонии (НМ).
