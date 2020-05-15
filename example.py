import pickle
import numpy as np
import scipy

import gabor_fitting

learned_dictionary = pickle.load(open('misc/learned_dictionary.p', 'rb'))
learned_dictionary = learned_dictionary.reshape((-1, 16, 16))  # 16x16 patches

fitted_gabors = np.zeros(learned_dictionary.shape)
for dict_elem in range(learned_dictionary.shape[0]):
  fitted_gabors[dict_elem] = gabor_fitting.fit(
      learned_dictionary[dict_elem], best_of=5)['reconstruction']
  if dict_elem != 0 and dict_elem % 20 == 0:
    print('Fitted ', dict_elem, 'dictionary elements')
