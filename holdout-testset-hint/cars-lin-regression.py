import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

use_holdout_testset = len(sys.argv) >= 2

# a fn to run the model on the holdout test set and print out the R^2 score:
def run_holdout_tests(holdout_testset_filename, model):
    # load the holdout test set into a pandas DataFrame:
    holdout_df = pd.read_csv(holdout_testset_filename)

    # Separate into a features matrix X_holdout and a target vector y_holdout:
    X_holdout = holdout_df.drop('price', axis=1)
    y_holdout = holdout_df['price']

    # Print out the R^2 score of the model on the holdout test set:
    print(f"\nR^2 score on holdout test set ({model.__class__.__name__}): {model.score(X_holdout, y_holdout)}")


# load the cars dataset into a pandas DataFrame:
cars_df = pd.read_csv('group1-cars.csv')

# Separate into a features matrix X and a target vector y:
X = cars_df.drop('price', axis=1)
y = cars_df['price']

# let's to a train/test split of the data fixing the random state for reproducibility:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Create a Linear Regression model and fit it to the training data:
lr = LinearRegression()
lr.fit(X_train, y_train)

# Print out the R^2 score of the model on the training & test data:
print(f"R^2 score on training data (linear): {lr.score(X_train, y_train)}")
print(f"R^2 score on test data (linear): {lr.score(X_test, y_test)}")

if use_holdout_testset:
    holdout_testset_filename = sys.argv[1]
    run_holdout_tests(holdout_testset_filename, lr)

