from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)

    print("Предсказания:", y_pred)
    print("Истинные значения:", y_test)
    print(f"Точность модели: {score:.2f}")
    print("-" * 50)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Task 1 — Загрузка данных и визуализация зависимостей
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    colors = ['red', 'green', 'blue']

    plt.figure(figsize=(5, 5))

    for i in range(3):
        plt.scatter(X[i * 50:(i + 1) * 50, 0], X[i * 50:(i + 1) * 50, 1], color=colors[i], label=target_names[i])
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('Sepal length vs Sepal width')
    plt.legend()
    plt.show()

    plt.figure(figsize=(5, 5))
    for i in range(3):
        plt.scatter(X[i * 50:(i + 1) * 50, 2], X[i * 50:(i + 1) * 50, 3], color=colors[i], label=target_names[i])
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.title('Petal length vs Petal width')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Task 2 — Seaborn Pairplot
    iris_df = pd.DataFrame(X, columns=iris.feature_names)
    iris_df['species'] = [target_names[i] for i in y]
    sns.pairplot(iris_df, hue='species')
    plt.show()

    # Task 3 — Создание 2 бинарных поднаборов данных
    # Набор 1: Setosa vs Versicolor (0 и 1)
    X_01 = X[0:100]
    y_01 = y[0:100]

    # Набор 2: Versicolor vs Virginica (1 и 2)
    X_12 = X[50:150]
    y_12 = y[50:150]

    # Task 4–8 — Обучение и оценка моделей на двух поднаборах
    print("Setosa vs Versicolor (Task 4–8)")
    train(X_01, y_01)

    print("Versicolor vs Virginica (Task 4–8)")
    train(X_12, y_12)

    # Task 9 — Генерация случайных данных и бинарная классификация
    X_rand, y_rand = make_classification(
        n_samples=1000, n_features=2, n_redundant=0,
        n_informative=2, random_state=1,
        n_clusters_per_class=1
    )

    plt.figure(figsize=(6, 5))
    plt.scatter(X_rand[:, 0], X_rand[:, 1], c=y_rand, cmap='coolwarm', edgecolor='k', alpha=0.6)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Случайно сгенерированные данные (Task 9)")
    plt.grid(True)
    plt.show()

    print("Классификация на случайных данных (Task 9)")
    train(X_rand, y_rand)
