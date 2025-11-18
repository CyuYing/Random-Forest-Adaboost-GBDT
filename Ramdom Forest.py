import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# 导入评估所需模块
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 配置 matplotlib 的环境
# 设置字体为黑体，解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

np.random.seed(0)


##############################################
# Some functions to plot our points and draw the lines
def plot_points(features, labels, fix_margins=True):
    X = np.array(features)
    y = np.array(labels)
    spam = X[np.argwhere(y == 1)]
    ham = X[np.argwhere(y == 0)]
    if fix_margins:
        plt.xlim(0, 11)
        plt.ylim(0, 11)
    plt.scatter([s[0][0] for s in spam],
                [s[0][1] for s in spam],
                s=100,
                color='cyan',
                edgecolor='k',
                marker='^')
    plt.scatter([s[0][0] for s in ham],
                [s[0][1] for s in ham],
                s=100,
                color='red',
                edgecolor='k',
                marker='s')
    plt.xlabel('Lottery')
    plt.ylabel('Sale')
    plt.legend(['Spam', 'Ham'])


def plot_model(X, y, model, fix_margins=True):
    X = np.array(X)
    y = np.array(y)
    plot_points(X, y)
    plot_step = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    if fix_margins:
        x_min = 0
        y_min = 0
        x_max = 12
        y_max = 12
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, colors=['red', 'blue'], alpha=0.2, levels=range(-1, 2))
    plt.contour(xx, yy, Z, colors='k', linewidths=3)
    plt.show()


def display_tree(dt):
    from six import StringIO
    from IPython.display import Image
    from sklearn.tree import export_graphviz
    import pydotplus
    dot_data = StringIO()
    export_graphviz(dt, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return Image(graph.create_png())

##################################################################
emails = np.array([
    [7, 8, 1],
    [3, 2, 0],
    [8, 4, 1],
    [2, 6, 0],
    [6, 5, 1],
    [9, 6, 1],
    [8, 5, 0],
    [7, 1, 0],
    [1, 9, 1],
    [4, 7, 0],
    [1, 3, 0],
    [3, 10, 1],
    [2, 2, 1],
    [9, 3, 0],
    [5, 3, 0],
    [10, 1, 0],
    [5, 9, 1],
    [10, 8, 1],
])
spam_dataset = pd.DataFrame(data=emails, columns=["Lottery", "Sale", "Spam"])
features = spam_dataset[['Lottery', 'Sale']]
labels = spam_dataset['Spam']

random_forest_classifier = RandomForestClassifier(random_state=0, n_estimators=5, max_depth=1)
random_forest_classifier.fit(features, labels)
print(random_forest_classifier.score(features, labels))

for dt in random_forest_classifier.estimators_:
    tree.plot_tree(dt, rounded=True)
    plt.show()
    plot_model(features, labels, dt)
    plt.show()

plot_model(features, labels, random_forest_classifier)