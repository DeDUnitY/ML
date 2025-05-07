import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(filename):
    data = pd.read_csv(filename)
    return data

def print_statistics(data):
    print("Статистика по данным:")
    print(f"Количество записей: {len(data)}")
    for column in data.columns:
        print(f"{column}: min={data[column].min()}, max={data[column].max()}, mean={data[column].mean():.2f}")

def linear_regression(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    a = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b = y_mean - a * x_mean
    return a, b

def plot_data(x, y):
    plt.figure(figsize=(14, 4))

    # Исходные данные
    plt.subplot(1, 3, 1)
    plt.scatter(x, y, color='blue')
    plt.title('Исходные точки')
    plt.xlabel('X')
    plt.ylabel('Y')

def plot_regression_line(x, y, a, b):
    plt.subplot(1, 3, 2)
    plt.scatter(x, y, color='blue', label='Данные')
    y_pred = a * x + b
    plt.plot(x, y_pred, color='red', label='Регрессионная прямая')
    plt.title('Регрессионная прямая')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

def plot_error_squares(x, y, a, b):
    plt.subplot(1, 3, 3)
    y_pred = a * x + b
    plt.scatter(x, y, color='blue', label='Данные')
    plt.plot(x, y_pred, color='red', label='Прямая')
    for xi, yi, ypi in zip(x, y, y_pred):
        plt.plot([xi, xi], [yi, ypi], 'g--')
        plt.fill_between([xi - 0.3, xi + 0.3], yi, ypi, color='green', alpha=0.2)
    plt.title('Квадраты ошибок')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

def main():
    filename = "student_scores.csv"
    data = load_data(filename)

    x_col = "Hours"
    y_col = "Scores"
    x = data[x_col].values
    y = data[y_col].values
    print_statistics(data[[x_col, y_col]])

    a, b = linear_regression(x, y)
    print(f"Коэффициенты регрессии: a={a:.2f}, b={b:.2f}")

    plot_data(x, y)
    plot_regression_line(x, y, a, b)
    plot_error_squares(x, y, a, b)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
