import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
dataframe = pd.read_csv('data.csv', delimiter=',', decimal=',', encoding='UTF-8')
dataframe.info()
print(dataframe.head(5))

dataframe.dropna(inplace=True)  # Видалення рядків з нульовими значеннями

dataframe['discounted_price(£)'] = dataframe['discounted_price(£)'].astype(float)
dataframe['original_price(£)'] = dataframe['original_price(£)'].astype(float)
dataframe['discount(%)'] = dataframe['discount(%)'].astype(float)
dataframe['discount(%)'] = dataframe['discount(%)'].abs()

dataframe.info()

sns.set(style='darkgrid')
plt.figure(figsize=(12, 6))

colors = sns.color_palette("Purples", len(dataframe['discount(%)'].unique()))

sns.countplot(x='discount(%)', data=dataframe, palette=colors)
plt.xlabel('Discount (%)')
plt.ylabel('Count')
plt.title('Distribution of Discounts')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

dataframe['rel_date'] = pd.to_datetime(dataframe['rel_date'], format='%Y-%m-%d')
dataframe['year'] = dataframe['rel_date'].dt.year.astype(float)
print(dataframe.head(5))

dataframe['discount_label'] = dataframe['discount(%)'].apply(lambda x: 0 if x < 30 else (1 if x < 50 else (2 if x == 50 else 3)))
dataframe.info()

data2 = dataframe.loc[:, ['discount(%)', 'year', 'original_price(£)', 'discounted_price(£)', 'discount_label']]
#--------------------------------------------------
print("-------------------------------------------")

sns.set(style='darkgrid')
plt.figure(figsize=(12, 6))

colors = sns.color_palette("Blues", len(data2['discount_label'].unique()))

sns.countplot(x='discount_label', data=data2, palette=colors)
plt.xlabel('Discount_label')
plt.ylabel('Count')
plt.title('Distribution of Discounts')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#---------------------------------------------
#Перший аналіз NKK

# Розділення даних на ознаки (X) та цільову змінну (y)
X = data2.drop(columns='discount_label')
y = data2['discount_label']

# Розділення на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)

# Ініціалізація KNN-класифікатора та параметрів для пошуку
classifier = KNeighborsClassifier()
parameters = {'n_neighbors': range(1, 25)}
grid_search = GridSearchCV(classifier, parameters, cv=10, verbose=1)
grid_search.fit(X_train, y_train)
print("Найкращі параметри: ", grid_search.best_estimator_)

# Використання найкращих параметрів для навчання моделі
best_classifier = grid_search.best_estimator_
best_classifier.fit(X_train, y_train)

# Прогнозування на тренувальній та тестовій вибірці
train_accuracy = round(best_classifier.score(X_train, y_train), 5)
test_accuracy = round(best_classifier.score(X_test, y_test), 5)

print("Точність KNN на тренувальній вибірці:", train_accuracy)
print("Точність KNN на тестовій вибірці:", test_accuracy)

plt.rcParams["figure.figsize"] = (15,4)
plt.gca().axes.get_yaxis().set_visible(False)
plt.plot(X_test.index, y_test, "yx", label = "True result")
plt.plot(X_test.index, best_classifier.predict(X_test), "b+", label = "Predict result")
plt.legend(loc='center right', shadow=True)
plt.show()

y_pred = best_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, cmap="Purples", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.grid(False)
plt.show()

compNKK1train = int(best_classifier.score(X_train, y_train)*100)
compNKK1test = int(best_classifier.score(X_test, y_test)*100)
KNN1 = best_classifier

print("-------------------------------------------")
#--------------------------------------------------
#аналіз 2 NKK

X = data2.drop(columns='discount(%)')
y = data2['discount(%)']

# Розділення на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)

# Ініціалізація KNN-класифікатора та параметрів для пошуку
classifier = KNeighborsClassifier()
parameters = {'n_neighbors': range(2, 25)}
grid_search = GridSearchCV(classifier, parameters, cv=10, verbose=1)
grid_search.fit(X_train, y_train)
print("Best parameters: ", grid_search.best_estimator_)

# Використання найкращих параметрів для навчання моделі
best_classifier = grid_search.best_estimator_
best_classifier.fit(X_train, y_train)

# Прогнозування на тренувальній та тестовій вибірці
train_accuracy = round(best_classifier.score(X_train, y_train), 5)
test_accuracy = round(best_classifier.score(X_test, y_test), 5)

print("Точність KNN на тренувальній вибірці:", train_accuracy)
print("Точність KNN на тестовій вибірці:", test_accuracy)

plt.rcParams["figure.figsize"] = (15,4)
plt.gca().axes.get_yaxis().set_visible(False)
plt.plot(X_test.index, y_test, "yx", label = "True result")
plt.plot(X_test.index, best_classifier.predict(X_test), "b+", label = "Predict result")
plt.legend(loc='center right', shadow=True)
plt.show()

y_pred = best_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, cmap="Purples", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.grid(False)
plt.show()

compNKK2train = int(best_classifier.score(X_train, y_train)*100)
compNKK2test = int(best_classifier.score(X_test, y_test)*100)
KNN2 = best_classifier

print("-------------------------------------------")
#----------------------------------------------------
#аналіз 1 DTC
X = data2.drop(columns='discount_label')
y = data2['discount_label']

# Розділення на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)

dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)

dtc_train_accuracy = round(dtc.score(X_train, y_train), 5)
dtc_test_accuracy = round(dtc.score(X_test, y_test), 5)

print("Точність DTC на тренувальній вибірці:", dtc_train_accuracy)
print("Точність DTC на тренувальній вибірці:", dtc_test_accuracy)

plt.rcParams["figure.figsize"] = (15,4)
plt.gca().axes.get_yaxis().set_visible(False)
plt.plot(X_test.index, y_test, "yx", label = "True result")
plt.plot(X_test.index,dtc.predict(X_test), "b+", label = "Predict result")
plt.legend(loc='center right', shadow=True)
plt.show()

y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.grid(False)
plt.show()

compCTD1train = int(dtc.score(X_train, y_train)*100)
compCTD1test = int(dtc.score(X_test, y_test)*100)
DTC1 = dtc

print("-------------------------------------------")
#--------------------------------------------------
#аналіз 2 DTC
X = data2.drop(columns='discount(%)')
y = data2['discount(%)']

# Розділення на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)

dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)

dtc_train_accuracy = round(dtc.score(X_train, y_train), 5)
dtc_test_accuracy = round(dtc.score(X_test, y_test), 5)

print("Точність DTC на тренувальній вибірці:", dtc_train_accuracy)
print("Точність DTC на тренувальній вибірці:", dtc_test_accuracy)

plt.rcParams["figure.figsize"] = (15,4)
plt.gca().axes.get_yaxis().set_visible(False)
plt.plot(X_test.index, y_test, "yx", label = "True result")
plt.plot(X_test.index,dtc.predict(X_test), "b+", label = "Predict result")
plt.legend(loc='center right', shadow=True)
plt.show()

y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.grid(False)
plt.show()
compCTD2train = int(dtc.score(X_train, y_train)*100)
compCTD2test = int(dtc.score(X_test, y_test)*100)
DTC2 = dtc

print("-------------------------------------------")
print("Аналіз 1")
print("Точність на тренувальній вибірці:")
print("KNN classifier: ", compNKK1train, '%,', 'DTC classifier: ', compCTD1train, '%')
print("Точність на тестовій вибірці:")
print("KNN classifier: ", compNKK1test, '%,', 'DTC classifier: ', compCTD1test, '%')

print("-------------------------------------------")
print("Аналіз 2")
print("Точність на тренувальній вибірці:")
print("KNN classifier: ", compNKK2train, '%,', 'DTC classifier: ', compCTD2train, '%')
print("Точність на тестовій вибірці:")
print("KNN classifier: ", compNKK2test, '%,', 'DTC classifier: ', compCTD2test, '%')


