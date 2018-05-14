import numpy
import scipy
import sys
import matplotlib
import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Check the versions of libraries
def check_versions():

# Python version
    print('Python: {}'.format(sys.version))
# scipy
    print('scipy: {}'.format(scipy.__version__))
# numpy
    print('numpy: {}'.format(numpy.__version__))
# matplotlib
    print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
    print('pandas: {}'.format(pandas.__version__))
# scikit-learn
    print('sklearn: {}'.format(sklearn.__version__))

if __name__ == "__main__":
    check_versions()

# Load dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pandas.read_csv(url, names=names)

# shape
#   print(dataset.shape)
# head
    print(dataset.head(20))
# descriptions
    print(dataset.describe())


