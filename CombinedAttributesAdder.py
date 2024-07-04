# Bjorn De Busschere, Bjorn.DeBusschere@student.uantwerpen.be, 3/04/2024
# =============================================================================
# Imports
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# =============================================================================
# Class
# =============================================================================


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Class to create an object to add new attributes to the California housing
    dataset bny combining other attributes. Atrributes that will be created are
    - rooms_per_household
    - bedrooms_per_room
    - population_per_household

    """
    def __init__(self):
        """
        Init method that does nothing
        """
        pass

    def fit(self, X, y=None):
        """
        fit method that returns self

        parameters:
            - X: should be the California housing dataset as either a panda
              DataFrame or an array.
        """
        return self

    def transform(self, X):
        """
        Method to create the now attributes and at them to the data

        parameters:
            - X: should be the California housing dataset as either a panda
              DataFrame or an array.
        """
        # Check if the provided data is in a DataFrame or array object
        if isinstance(X, pd.DataFrame):
            # Creating the new attributes
            rooms_per_household = X['total_rooms']/X['households']
            bedrooms_per_room = X['total_bedrooms']/X['total_rooms']
            population_per_household = X['population']/X['households']
            # Adding them to the data
            data = np.c_[X.values, rooms_per_household, bedrooms_per_room,
                         population_per_household]
            header = list(X.columns) + ['rooms_per_household',
                                        'bedrooms_per_room',
                                        'population_per_household']
            return pd.DataFrame(data, columns=header)
        elif isinstance(X, np.ndarray):
            # Creating the new attributes
            rooms_per_household = X[:, 3]/X[:, 6]
            bedrooms_per_room = X[:, 4]/X[:, 3]
            population_per_household = X[:, 5]/X[:, 6]
            # Adding them to the data
            data = np.c_[X, rooms_per_household, bedrooms_per_room,
                         population_per_household]
            return data
        else:
            #Raising an error
            raise TypeError("X must be a pd.DataFrame or np.ndarray.")
