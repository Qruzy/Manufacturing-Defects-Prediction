import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

def plot_defect_distribution(data):
    """Plot bar and pie chart of defect status distribution."""
    fig, axes = plt.subplots(ncols=2, figsize=(15, 6))
    plt.suptitle('Total of Manufacturing Defects')

    # Plotting bar chart
    count = data["DefectStatus"].value_counts()
    count.plot(kind="bar", ax=axes[0])
    axes[0].set_ylabel("Count")
    axes[0].set_xlabel("Defect Status")
    for container in axes[0].containers:
        axes[0].bar_label(container)

    # Plotting pie chart
    count.plot(kind="pie", ax=axes[1], autopct="%0.2f%%")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("")

    plt.show()
    return fig

def plot_correlation_matrix(data):
    """Plot correlation heatmap of the dataset."""
    fig, ax = plt.subplots(figsize=(15, 6))

    # Encode 'DefectStatus' if it's categorical for the correlation matrix
    data_encoded = data.copy()
    if data_encoded["DefectStatus"].dtype == "object":
        le = LabelEncoder()
        data_encoded['DefectStatus'] = le.fit_transform(data_encoded['DefectStatus'])
    
    # Plot heatmap
    matrix = data_encoded.corr()
    sns.heatmap(matrix, annot=True, ax=ax)
    plt.show()
    return fig

def plot_feature_importance(data):
    """Train a RandomForest model and plot feature importance."""
    fig, ax = plt.subplots(figsize=(15, 6))

    # Data preprocessing
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Encode categorical target if needed
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    # Handling class imbalance
    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Model training
    forest = RandomForestClassifier()
    forest.fit(X_train, y_train)

    # Plotting feature importances
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    feature_names = data.columns[:-1]
    feature_importances = pd.Series(importances, index=feature_names)
    
    feature_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature Importances using Mean Decrease in Impurity (MDI)")
    ax.set_ylabel("Mean decrease in impurity")
    plt.tight_layout()

    return fig

def plot_graph(data):
    """Generate all plots for data exploration and feature importance."""
    fig1 = plot_defect_distribution(data)
    fig2 = plot_correlation_matrix(data)
    fig3 = plot_feature_importance(data)
    return fig1, fig2, fig3
