import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Загрузка набора данных diabetes
diabetes = datasets.load_diabetes()

# Преобразуем в DataFrame
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

print("Первые строки набора данных:")
print(df.head())


# Выбираем один признак для простоты (например, 'bmi')
feature = 'bmi'
X = df[[feature]].values
y = df['target'].values

# ====== Линейная регрессия с помощью sklearn ======
model = linear_model.LinearRegression()
model.fit(X, y)
y_pred_sklearn = model.predict(X)
a_sklearn = model.coef_[0]
b_sklearn = model.intercept_

print(f"\nsklearn коэффициенты: a = {a_sklearn:.2f}, b = {b_sklearn:.2f}")

# ====== Линейная регрессия вручную ======
x_mean = np.mean(X)
y_mean = np.mean(y)
a_manual = np.sum((X - x_mean) * (y.reshape(-1, 1) - y_mean)) / np.sum((X - x_mean)**2)
b_manual = y_mean - a_manual * x_mean
y_pred_manual = a_manual * X + b_manual

print(f"Мои коэффициенты: a = {a_manual:.2f}, b = {b_manual:.2f}")

# ====== График ======
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='gray', label='Данные')
plt.plot(X, y_pred_sklearn, color='red', label='Sklearn')
plt.plot(X, y_pred_manual, color='blue', linestyle='--', label='Ручной метод')
plt.xlabel(feature)
plt.ylabel('target')
plt.title(f'Линейная регрессия по признаку "{feature}"')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ====== Таблица с предсказаниями ======
results = pd.DataFrame({
    feature: X.flatten(),
    'Фактическое значение': y,
    'Предсказание sklearn': y_pred_sklearn,
    'Предсказание вручную': y_pred_manual.flatten()
})

print("\nТаблица предсказаний (первые 10 строк):")
print(results.head(10).round(2))

mae = mean_absolute_error(y, y_pred_sklearn)
r2 = r2_score(y, y_pred_sklearn)
mape = mean_absolute_percentage_error(y, y_pred_sklearn)

print("\nМетрики качества модели (sklearn):")
print(f"MAE (средняя абсолютная ошибка): {mae:.2f}")
print(f"R² (коэффициент детерминации): {r2:.2f}")
print(f"MAPE (средняя абсолютная процентная ошибка): {mape * 100:.2f}%")