# packages
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# set seed
seed = 314

def train_model(X, y, seed=seed):
    """
    Build a GBM on given data
    """
    model = GradientBoostingClassifier(
        learning_rate=0.05,     #slower
        n_estimators=300,   #more trees
        max_depth=4,    #shallow twee... twea??
        subsample=0.8,      #i dunno i guessed.
        min_samples_leaf=10,    #more predictions
        random_state=0
    )
    model.fit(X, y)
    return model