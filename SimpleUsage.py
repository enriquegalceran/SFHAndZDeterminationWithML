import keras
import numpy as np

# Reshape the input spectra to (n_datapoints, spectra_length, 1)
input_spectra = np.reshape(n_datapoints, 1761, 1)
# Reshape the input photometry to (n_datapoints, n_filters)
input_photometry = np.reshape(n_datapoints, 5)
cnn = keras.models.load_model("path/to/trained_cnn.h5")
predicted_sfh, predicted_Z = cnn.predict([input_spectra, input_photometry])
