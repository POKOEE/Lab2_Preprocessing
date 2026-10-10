import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1. Загрузка данных
# ------------------------------------------------------------
print("=" * 60)
print("ЗАДАНИЕ 3. ЗАГРУЗКА ДАННЫХ")
print("=" * 60)

try:
    df3 = sns.load_dataset('titanic')
except Exception:
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    df3 = pd.read_csv(url)

print(f"Размер датасета ДО удаления пропусков в 'age': {df3.shape}")
print("\nПропуски в столбцах:")
print(df3.isnull().sum())

# ------------------------------------------------------------
# 2. УДАЛЕНИЕ строк с пропусками в 'age' (вместо заполнения)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("ШАГ 2. УДАЛЕНИЕ СТРОК С ПРОПУСКАМИ В 'age'")
print("=" * 60)

initial_size = df3.shape[0]
df3 = df3.dropna(subset=['age'])
final_size = df3.shape[0]

print(f"Размер до удаления: {initial_size} строк")
print(f"Размер после удаления: {final_size} строк")
print(f"Удалено строк: {initial_size - final_size}")
print(f"Потеря данных: {(initial_size - final_size) / initial_size * 100:.2f}%")

# embarked — заполняем модой (единственный столбец с пропусками)
mode_embarked = df3['embarked'].mode()[0]
df3['embarked'] = df3['embarked'].fillna(mode_embarked)
print(f"\nembarked заполнен модой = {mode_embarked}")

# Удаление столбцов
df3.drop(['deck', 'embark_town'], axis=1, inplace=True)
print("Столбцы 'deck' и 'embark_town' удалены")

print("\nПропуски после обработки:")
print(df3.isnull().sum())

# ------------------------------------------------------------
# 3. One-Hot Encoding для sex и embarked
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("ШАГ 3. ONE-HOT ENCODING")
print("=" * 60)

df3 = pd.get_dummies(df3, columns=['sex', 'embarked'], drop_first=True)

# Удаление лишних столбцов
df3.drop(['pclass', 'who', 'adult_male', 'class', 'alive', 'alone'],
         axis=1, inplace=True)

print("\nДатасет после One-Hot Encoding (первые 5 строк):")
print(df3.head())
print(f"\nСтолбцы: {list(df3.columns)}")

# ------------------------------------------------------------
# 4. Стандартизация признака fare (StandardScaler)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("ШАГ 4. СТАНДАРТИЗАЦИЯ 'fare'")
print("=" * 60)

print("\nСтатистика 'fare' ДО стандартизации:")
print(df3['fare'].describe())

scaler = StandardScaler()
df3['fare'] = scaler.fit_transform(df3[['fare']])

print("\nСтатистика 'fare' ПОСЛЕ стандартизации:")
print(df3['fare'].describe())

print("\nДатасет после стандартизации (первые 5 строк):")
print(df3.head())

# ------------------------------------------------------------
# 5. УДАЛЕНИЕ выбросов в 'fare' по методу IQR
#    (именно удаление строк, а не capping!)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("ШАГ 5. УДАЛЕНИЕ ВЫБРОСОВ В 'fare' (метод IQR)")
print("=" * 60)

# Визуализация ДО удаления
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=df3['fare'])
plt.title('Boxplot для Fare (ДО удаления выбросов)')

Q1 = df3['fare'].quantile(0.25)
Q3 = df3['fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nQ1 = {Q1:.3f}")
print(f"Q3 = {Q3:.3f}")
print(f"IQR = {IQR:.3f}")
print(f"Нижняя граница = {lower_bound:.3f}")
print(f"Верхняя граница = {upper_bound:.3f}")

before_outliers = df3.shape[0]
df3 = df3[(df3['fare'] >= lower_bound) & (df3['fare'] <= upper_bound)]
after_outliers = df3.shape[0]

print(f"\nСтрок ДО удаления выбросов: {before_outliers}")
print(f"Строк ПОСЛЕ удаления выбросов: {after_outliers}")
print(f"Удалено выбросов: {before_outliers - after_outliers}")

# Визуализация ПОСЛЕ удаления
plt.subplot(1, 2, 2)
sns.boxplot(y=df3['fare'])
plt.title('Boxplot для Fare (ПОСЛЕ удаления выбросов)')

plt.tight_layout()
plt.savefig('task3_boxplot.png', dpi=150)
plt.show()

print("\nГрафики сохранены в файл 'task3_boxplot.png'")

# ------------------------------------------------------------
# 6. Сравнение подходов к обработке пропусков
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("СРАВНЕНИЕ ПОДХОДОВ К ОБРАБОТКЕ ПРОПУСКОВ В 'age'")
print("=" * 60)

loss_percentage = (initial_size - final_size) / initial_size * 100

print(f"\nИсходный размер датасета: {initial_size} строк")
print(f"Задание 1 (заполнение медианой): 891 строка (100%)")
print(f"Задание 2 (заполнение средним):  891 строка (100%)")
print(f"Задание 3 (удаление строк):      {final_size} строк "
      f"({final_size / initial_size * 100:.2f}%)")

print(f"\nПотеря данных при удалении строк: {loss_percentage:.2f}%")
print(f"Сохранено при заполнении: 100.00%")

# ------------------------------------------------------------
# 7. Финальный вывод
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("ВЫВОД ПО ЗАДАНИЮ 3")
print("=" * 60)
print("\nУдаление строк с пропусками привело к большей потере данных:")
print(f"- Потеряно {initial_size - final_size} строк из {initial_size}")
print(f"- Это {loss_percentage:.2f}% исходной выборки")
print()
print("Заполнение медианой (Задание 1) и средним (Задание 2) сохраняет")
print("все 891 строку, но вносит небольшую погрешность в данные.")
print()
print("Когда что использовать:")
print("- УДАЛЕНИЕ строк — если пропусков мало (<5%) или они случайны.")
print("- ЗАПОЛНЕНИЕ — если пропусков много (>10%), чтобы не терять данные.")
print()
print("В нашем случае пропуски в 'age' составляют ~19.87% —")
print("это много, поэтому заполнение предпочтительнее удаления.")

# ------------------------------------------------------------
# 8. Финальный вид датасета
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("ФИНАЛЬНЫЙ ДАТАСЕТ (Задание 3):")
print("=" * 60)
print(df3.head())
print(f"\nРазмер итогового датасета: {df3.shape}")

print("\nСтатистика финального датасета:")
print(df3.describe())
