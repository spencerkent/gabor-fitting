"""
A utility for fitting gabor functions to greyscale image dictionaries.

Useful in the context of dictionary learning to find a parameterization of
learned dictionary elements. Inspired by implementations from Dylan Paiton and
Jesse Livezey.

Spencer Kent, May 2020
"""
import numpy as np
import scipy.signal
import scipy.optimize

def make_gabor(patch_size, gabor_parameters, return_separate_env_grating=False):
  """
  Generate an image patch containing a 2D Gabor function.

  Here is my convention for parameterizing a Gabor, which may be slightly
  different than what you've see online or elsewhere.
  1) There are two orthogonal axes for a Gabor, the direction
     (aligned/parallel) to the grating is the primary axis. It is also the axis
     along which the envelope is larger. The direction normal to the grating
     is the secondary axis.
  2) Orientation is in *radians counter-clockwise from horizontal* of the
     *primary* axis. At 0 radians, the filter is most sensitive to
     horizontal lines. At pi/2 it is most sensitive to vertical lines.
  3) The envelope width is the standard deviation \sigma of the gaussian
     envelope along the primary axis
  4) The envelope aspect ratio is \sigma along the secondary axis divided by
     \sigma along the primary axis. It should be in the interval [0, 1].

  Parameters
  ----------
  patch_size : (int, int)
      The spatial size of the image patch containing the Gabor, in pixels
  gabor_parameters : dictionary
      'position_yx' : (int, int)
        The position, in pixels, of the Gabor, relative to the center of the
        patch. This uses "array-indexing" rather than "Cartesian" indexing
        so increasing y is *down*.
      'orientation' : float
        The orientation of the Gabor, in radians
      'frequency' : float
        Frequency, in cycles / pixel.
      'phase' : float
        Phase offest in radians (sin() used for grating).
      'envelope_width' : float
        The standard deviation of the gaussian envelope along the primary axis
      'envelope_aspect' : float
        The ratio of envelope size between the secondary and primary axes.
  return_separate_env_grating : bool, optional
      Along with the Gabor, return the Gaussian envelop and sine-wave grating
      that were used to generate it. Useful for debugging purposes.
      Default False.

  Returns
  -------
  gabor : ndarray(float32, size=(patch_size[0], patch_size[1]))
      The Gabor function patch. *I normalize this to have an l2 norm of 1.0*

  Example
  -------
  make_gabor(patch_size=(16, 16), gabor_parameters={
    'position_yx'=(3, 2), 'orientation': np.pi/6, 'frequency': 1/4,
    'phase'=0, 'envelope_width'=3, 'envelope_aspect': 0.5)
  """
  assert patch_size[0] > 2 and patch_size[1] > 2
  assert gabor_parameters['envelope_aspect'] <= 1.0
  assert gabor_parameters['envelope_aspect'] > 0.0
  v_coords, h_coords = _get_coords(patch_size[0], patch_size[1])

  mv, mh = np.meshgrid(v_coords, h_coords, indexing='ij')
  mv_trans, mh_trans = np.meshgrid(
      gabor_parameters['position_yx'][0] * np.ones(patch_size[0]),
      gabor_parameters['position_yx'][1] * np.ones(patch_size[1]),
      indexing='ij')
  mh_prime = ((mh-mh_trans) * np.cos(gabor_parameters['orientation']) -
              (mv-mv_trans) * np.sin(gabor_parameters['orientation']))
  mv_prime = ((mh-mh_trans) * np.sin(gabor_parameters['orientation']) +
              (mv-mv_trans) * np.cos(gabor_parameters['orientation']))
  envelope = np.exp(-1 * (
    (mh_prime**2) + ((mv_prime / gabor_parameters['envelope_aspect'])**2))
    / (2*(gabor_parameters['envelope_width']**2)))
  grating = np.sin(2 * np.pi * gabor_parameters['frequency'] * mv_prime +
                   gabor_parameters['phase'])
  gabor = envelope * grating
  if return_separate_env_grating:
    return gabor / np.linalg.norm(gabor), envelope, grating
  else:
    return gabor / np.linalg.norm(gabor)


def fit(dictionary_element_2d, allowed_retries=5, best_of=1):
  """
  This routine fits a 2D gabor function to dictionary_element_2d

  It uses a fairly specific sequence of different curve fits based on SciPy's
  nonlinear least squares curve-fitting package. First we estimate the
  envelope of the gabor, then the spatial frequency of the grating, then the
  orientation and phase of the grating, and finally we go back and fine tune
  some of the parameters. We get the option of taking the best fit from an
  ensemble--recommended for the most robust performance.

  Parameters
  ----------
  dictionary_element_2d : ndarray(size=(kh, kw))
      The dictionary element, with height kh and width kw. Currently only
      works for greyscale images.
  allowed_retries : int, optional
      The estimate is based on a nonlinear least-squares regression implemented
      in SciPy that can sometimes fail to converge. This indicates the maximum
      number of times we're allowed to retry the fit with different initial
      conditions before giving up. Default 5
  best_of : int, optional
      We will generate this many different fits for the dictionary_element_2d,
      taking the one which matches best. Default 1

  Returns
  -------
  best_fit : dictionary
      Gives the best-fitting gabor function for dictionary_element_2d.
      'parameters' : dictionary
        Parameters of the Gabor function. Can be used with make_gabor()
      'reconstruction' : ndarray(size=(kh, kw))
        For convenience, the output of make_gabor() on these parameters, the
        reconstruction from the fit.
      'error' : float
        The l2-norm of the reconstruction error
  """
  if np.abs(np.linalg.norm(dictionary_element_2d) - 1.0) > 1e-5:
    print('I have only tested this for gabor functions that are unit-l2-norm.',
          'Proceed with caution')
  de_v_coords, de_h_coords = _get_coords(
      dictionary_element_2d.shape[0], dictionary_element_2d.shape[1])
  mesh_de_coords = np.meshgrid(de_v_coords, de_h_coords, indexing='ij')

  error_msgs = []
  fitted_gabors = []
  for trial_idx in range(best_of):
    # get an initial estimate of the envelope
    fitted_env_params, fitted_env = fit_envelope(
        dictionary_element_2d, allowed_retries=allowed_retries)
    if 'fit failed to converge' in fitted_env_params:
      error_msgs.append('I had difficulty with the first envelope fit')
      continue
    # estimate the spatial frequency. We won't change this further.
    spatial_freq = infer_spatial_frequency(dictionary_element_2d,
        fitted_env_params, allowed_retries=allowed_retries)
    if spatial_freq < 0:
      error_msgs.append('I had difficulty estimating the spatial frequency')
      continue

    # fit the phase of the grating and fine-tune the orientation
    fixed = (mesh_de_coords + [fitted_env] +
             [fitted_env_params['position_yx'][0]*np.ones(fitted_env.shape),
              fitted_env_params['position_yx'][1]*np.ones(fitted_env.shape),
              spatial_freq*np.ones(fitted_env.shape)])
    for fit_attempt in range(allowed_retries):
      initial_guess = (fitted_env_params['orientation'],
                       np.random.uniform(low=-np.pi, high=np.pi))
      try:
        popt, pcov = scipy.optimize.curve_fit(
            _fit_grating_orientation_and_phase, fixed,
            dictionary_element_2d.ravel(), p0=initial_guess,
            bounds=([fitted_env_params['orientation'] - (np.pi/6), -np.pi],
                    [fitted_env_params['orientation'] + (np.pi/6), np.pi]))
      except:
        pass  # try again
      else:
        break
    else:
      error_msgs.append('I had difficulty in fitting phase and ' +
                        'finetuning the orientation')
      continue
    # For the purposes of less-ambiguous comparison between different gabors
    # I enforce a certain convention on orientation and phase.
    popt[0], flipped_flag = _standardize_env_orientation(popt[0])
    if flipped_flag:
      popt[1] = popt[1] + np.pi
    popt[1] = _standardize_phase(popt[1])

    current_best_guess = {
        'orientation': popt[0],  # more accurate than initial envelope est.
        'envelope_width': fitted_env_params['envelope_width'],
        'envelope_aspect': fitted_env_params['envelope_aspect'],
        'frequency': spatial_freq,
        'phase': popt[1],
        'position_yx': fitted_env_params['position_yx']}

    # the initial estimate for the envelope is sometimes overzealous and we
    # can finetune it here. If we allow the envelope to translate, this
    # tends to cause problems, so we just modify width, aspect, and magnitude,
    # which are all somewhat coupled.
    inferred_grating = _grating(mesh_de_coords,
        current_best_guess['position_yx'][0],
        current_best_guess['position_yx'][1],
        current_best_guess['orientation'],
        current_best_guess['frequency'],
        current_best_guess['phase']).reshape(dictionary_element_2d.shape)
    fixed = mesh_de_coords + [
        current_best_guess['position_yx'][0]*np.ones(dictionary_element_2d.shape),
        current_best_guess['position_yx'][1]*np.ones(dictionary_element_2d.shape),
        current_best_guess['orientation']*np.ones(dictionary_element_2d.shape),
        inferred_grating]
    for fit_attempt in range(allowed_retries):
      initial_guess = (current_best_guess['envelope_width'],
                       current_best_guess['envelope_aspect'],
                       1.25*fitted_env_params['magnitude'])
      try:
        popt, pcov = scipy.optimize.curve_fit(
            _fit_envelope_width_aspect, fixed,
            dictionary_element_2d.ravel(), p0=initial_guess,
            bounds=([current_best_guess['envelope_width'] * 0.5,
                     current_best_guess['envelope_aspect'] * 0.75,
                     fitted_env_params['magnitude']],
                    [current_best_guess['envelope_width'] * 1.5,
                     min(current_best_guess['envelope_aspect'] * 1.25, 1.0),
                     fitted_env_params['magnitude']*1.5]))
      except:
        pass  # try again
      else:
        break
    else:
      error_msgs.append('I had difficulty in finetuning the envelope')
      continue
    current_best_guess['envelope_width'] = popt[0]
    current_best_guess['envelope_aspect'] = popt[1]

    # One thing that makes gabor fitting hairy is that, among the several
    # highly-coupled and redundant variables, phase shifts can be induced by
    # just about everything...including what we just did to the envelope.
    # We'll try and correct for it here.
    env_finetuned = _fit_gaussian(mesh_de_coords,
        current_best_guess['position_yx'][0],
        current_best_guess['position_yx'][1],
        current_best_guess['orientation'],
        current_best_guess['envelope_width'],
        current_best_guess['envelope_aspect'],
        popt[2]).reshape(dictionary_element_2d.shape)
    fixed = (mesh_de_coords + [env_finetuned] + [
      current_best_guess['position_yx'][0]*np.ones(dictionary_element_2d.shape),
      current_best_guess['position_yx'][1]*np.ones(dictionary_element_2d.shape),
      current_best_guess['frequency']*np.ones(dictionary_element_2d.shape),
      current_best_guess['orientation']*np.ones(dictionary_element_2d.shape)])
    for fit_attempt in range(allowed_retries):
      initial_guess = (current_best_guess['phase'])
      try:
        popt, pcov = scipy.optimize.curve_fit(
            _fit_grating_phase, fixed, dictionary_element_2d.ravel(),
            p0=initial_guess, bounds=([-np.pi], [np.pi]))
      except:
        pass  # try again
      else:
        break
    else:
      error_msgs.append('I had difficulty in fine-tuning the phase')
      continue
    popt[0] = _standardize_phase(popt[0])
    current_best_guess['phase'] = popt[0]

    # That's it, whew!
    fitted_gabors.append({
      'parameters': current_best_guess,
      'reconstruction': make_gabor(dictionary_element_2d.shape,
                                   current_best_guess)})
  # end of trials
  if len(fitted_gabors) == 0:
    raise RuntimeError('Was not able to fit this dictionary element.\n' +
                       'The error message for each trial was:\n' +
                       str(error_msgs))
  else:
    recon_errors = [np.linalg.norm(fitted_gabors[x]['reconstruction'] -
                    dictionary_element_2d) for x in range(len(fitted_gabors))]
    winner = np.argmin(recon_errors)
    best_fit = fitted_gabors[winner]
    best_fit['error'] = recon_errors[winner]
    return best_fit


def fit_envelope(dictionary_element_2d, allowed_retries=5):
  """
  Estimates the envelope of dictionary_element_2d with a 2D gaussian

  Parameters
  ----------
  dictionary_element_2d : ndarray(size=(kh, kw))
      The dictionary element, with height kh and width kw. Currently only
      works for greyscale images.
  allowed_retries : int, optional
      The fit is based on a nonlinear least-squares regression implemented
      in SciPy that can sometimes fail to converge. This indicates the maximum
      number of times we're allowed to retry the fit with different initial
      conditions before giving up. Default 5

  Returns
  -------
  env_params : dictionary
      Specifies the 6 fitted parameters for the gaussian envelope. These are
      'position_yx' : position of envelope center
      'orientation' : radians from the horizontal of the *primary* axis
      'envelope_width' : the standard deviation of the gaussian, along the
        primary axis
      'envelope_aspect' : aspect ratio of the envelope, the ratio of the
        secondary axis width to the primary axis width. This is in the
        interval [0, 1]
      'magnitude' : The inferred magnitude of the envelope. This gives an
        extra degree of freedom, for instance if the dictionary element has
        been renormalized.
  env_estimated : ndarray(size=(kh, kw))
      The estimated envelope of the dictionary element.
  """
  # Use the Hilbert Transform to estimate the envelope
  env_est = np.abs(scipy.signal.hilbert(dictionary_element_2d))
  # ^there's a "2d" version of this in scipy but I've found this works better
  de_v_coords, de_h_coords = _get_coords(env_est.shape[0], env_est.shape[1])
  # find the approximate center of the envelope
  temp = np.unravel_index(np.argsort(env_est.ravel()), env_est.shape)
  approx_center = (np.mean(de_v_coords[temp[0][-4:]]),
                   np.mean(de_h_coords[temp[1][-4:]]))
  approx_max = env_est[temp[0][-1], temp[1][-1]]

  # fit a 2d gaussian function to this signal
  mesh_de_coords = np.meshgrid(de_v_coords, de_h_coords, indexing='ij')
  for fit_attempt in range(allowed_retries):
    initial_guess = (approx_center[0], approx_center[1],
                     np.random.uniform(low=0.0, high=np.pi),
                     np.random.uniform(low=2.0, high=6.0),
                     0.5, approx_max)
    # ^pos_y, pos_x, orientation, std, aspect, magnitude
    try:
      popt, pcov = scipy.optimize.curve_fit(
          _fit_gaussian, mesh_de_coords, env_est.ravel(), p0=initial_guess,
          bounds=([de_v_coords[0]*1.5, de_h_coords[0]*1.5, -np.inf,
                   0.0, 0.0, 0.0],
                  [de_v_coords[-1]*1.5, de_h_coords[-1]*1.5, np.inf,
                   env_est.shape[0]/2, 1.0, np.inf]))
      # ^bounds help to resolve ambiguity but also allow enough deg. of freedom
    except:
      pass  # try again
    else:
      break
  else:
    # didn't work, the least we can do is return something sensible
    return {'fit failed to converge'}, np.zeros(dictionary_element_2d.shape)
  popt[2], _ = _standardize_env_orientation(popt[2])
  env_est_fitted = _fit_gaussian(mesh_de_coords, *popt)
  env_params = {'position_yx': (popt[0], popt[1]),
                'orientation': popt[2],
                'envelope_width': popt[3],
                'envelope_aspect': popt[4],
                'magnitude': popt[5]}
  return env_params, env_est_fitted.reshape(env_est.shape)


def infer_spatial_frequency(dictionary_element_2d, envelope_params,
                            allowed_retries=5):
  """
  This uses the 2D power spectrum to estimate spatial frequency.

  It requires that you first estimate the envelope of the patch using the
  fit_envelope() function in this file. The method below can work pretty well
  for the typical gabor, including those which overhang the patch edge
  substantially.

  Parameters
  ----------
  dictionary_element_2d : ndarray(size=(kh, kw))
      The dictionary element, with height kh and width kw. Currently only
      works for greyscale images.
  envelope_params : dictionary
      See the docstring for fit_envelope()
  allowed_retries : int, optional
      The estimate is based on a nonlinear least-squares regression implemented
      in SciPy that can sometimes fail to converge. This indicates the maximum
      number of times we're allowed to retry the fit with different initial
      conditions before giving up. Default 5.

  Returns
  -------
  estimated_spatial_freq : float
    The estimated spatial frequency, in cycles per pixel
  """
  # compute an upsampled DFT of the dictionary element
  upsampling_factor = 4
  dft_nsamps = np.array(dictionary_element_2d.shape)*upsampling_factor
  # power_spectrum in shifted 2d frequency coords
  power_spectrum = np.fft.fftshift(
      np.abs(np.fft.fft2(dictionary_element_2d, dft_nsamps))**2)
  freq_coords_v = np.fft.fftshift(np.fft.fftfreq(dft_nsamps[0]))
  freq_coords_h = np.fft.fftshift(np.fft.fftfreq(dft_nsamps[1]))
  exact_zero = (np.where((freq_coords_v == 0.0))[0][0],
                np.where((freq_coords_h == 0.0))[0][0])
  # mask out one of the lobes to make fitting with single gaussian easier
  # visualize power_spectrum and fit_this to see what we're doing here
  fit_this = np.copy(power_spectrum)
  if (envelope_params['orientation'] <= np.pi/4 or
      ((envelope_params['orientation'] > 3*np.pi/4) and
       (envelope_params['orientation'] <= np.pi))):
    # Vertically dominant lobes, mask the bottom half
    fit_this[exact_zero[0]+1:, :] = 0.0
  else:
    # Horizontally dominant lobes, mask the right half
    fit_this[:, exact_zero[1]+1:] = 0.0

  # find the approximate center of the envelope
  temp = np.unravel_index(np.argsort(fit_this.ravel()), fit_this.shape)
  approx_center = (np.mean(freq_coords_v[temp[0][-4:]]),
                   np.mean(freq_coords_h[temp[1][-4:]]))
  approx_max = fit_this[temp[0][-1], temp[1][-1]]
  freq_coords = np.meshgrid(freq_coords_v, freq_coords_h, indexing='ij')

  # fit a 2d gaussian function to our masked power spectrum
  for fit_attempt in range(allowed_retries):
    initial_guess = (approx_center[0], approx_center[1],
                     np.random.uniform(low=0.0, high=np.pi),
                     np.random.uniform(low=1/(2*np.pi*6.0),  # spatial env 6pix
                                       high=1/(2*np.pi*2.0)), # spatial env 2pix
                     0.5, approx_max)
    # ^pos_y, pos_x, orientation, std, aspect, magnitude
    try:
      popt, pcov = scipy.optimize.curve_fit(
          _fit_gaussian, freq_coords, fit_this.ravel(), p0=initial_guess,
          bounds=([-0.7, -0.7, -np.inf, 0.0, 0.0, 0.0],
                  [0.7, 0.7, np.inf, 1/(2*np.pi*1.0), 1.0, np.inf]))
      # ^bounds help to resolve ambiguity but also allow enough deg. of freedom
    except:
      pass  # try again
    else:
      break
  else:
    # didn't work, the least we can do is return something sensible
    return -np.inf
  # we could use some mixture of approx_center and the inferred gaussian
  # center, but in my experience the gaussian is closer, more robust
  estimated_spatial_freq = (popt[0]**2 + popt[1]**2)**0.5
  return estimated_spatial_freq


######################################################################
# The following functions are used by the SciPy curve-fitting module
# Because of the API, fixed parameters are passed in as a tuple to the
# first argument, and the other arguments will be fitted.
######################################################################

# 2D-gaussian with shift from (0, 0), specified by pos_y and pos_x.
def _fit_gaussian(fixed_params, pos_y, pos_x, orientation, std, aspect, magnitude):
  (y, x) = fixed_params
  xprime = ((x - pos_x) * np.cos(orientation) -
            (y - pos_y) * np.sin(orientation))
  yprime = ((x - pos_x) * np.sin(orientation) +
            (y - pos_y) * np.cos(orientation))
  return magnitude*np.exp(
      -1 * ((xprime**2) + ((yprime / aspect)**2)) / (2*(std**2))).ravel()

# 2D-grating with a shift from (0, 0) specified by pos_y and pos_x.
def _grating(fixed_params, pos_y, pos_x, orientation, frequency, phase):
  (y, x) = fixed_params
  xprime = ((x - pos_x) * np.cos(orientation) -
            (y - pos_y) * np.sin(orientation))
  yprime = ((x - pos_x) * np.sin(orientation) +
            (y - pos_y) * np.cos(orientation))
  return np.sin((2 * np.pi * frequency * yprime) + phase).ravel()

# Grating, but underneath an existing envelope and with a predetermined freq.
def _fit_grating_orientation_and_phase(fixed_params, orientation, phase):
  (y, x, precomputed_env, env_pos_y, env_pos_x, frequency) = fixed_params
  xprime = ((x - env_pos_x) * np.cos(orientation) -
            (y - env_pos_y) * np.sin(orientation))
  yprime = ((x - env_pos_x) * np.sin(orientation) +
            (y - env_pos_y) * np.cos(orientation))
  grating = np.sin((2 * np.pi * frequency * yprime) + phase)
  return (precomputed_env * grating).ravel()

# Fit the phase of a grating under and env, all other parameters held fixed
def _fit_grating_phase(fixed_params, phase):
  (y, x, precomputed_env, env_pos_y, env_pos_x,
   frequency, orientation) = fixed_params
  xprime = ((x - env_pos_x) * np.cos(orientation) -
            (y - env_pos_y) * np.sin(orientation))
  yprime = ((x - env_pos_x) * np.sin(orientation) +
            (y - env_pos_y) * np.cos(orientation))
  grating = np.sin((2 * np.pi * frequency * yprime) + phase)
  return (precomputed_env * grating).ravel()

# fit the width and aspect ratio (and magnitude) of Gabor envelope. Fine-tuning
def _fit_envelope_width_aspect(fixed_params, std, aspect, magnitude):
  (y, x, pos_y, pos_x, orientation, precomputed_grating) = fixed_params
  xprime = ((x - pos_x) * np.cos(orientation) -
            (y - pos_y) * np.sin(orientation))
  yprime = ((x - pos_x) * np.sin(orientation) +
            (y - pos_y) * np.cos(orientation))
  envelope = magnitude*np.exp(
      -1 * ((xprime**2) + ((yprime / aspect)**2)) / (2*(std**2)))
  return (envelope * precomputed_grating).ravel()


#########################################################
# Just a few (non-curve-fitting related) helper functions
#########################################################

def _get_coords(vert_size, horz_size):
  # just our convention for labeling position in a 2d image. The middle
  # pixel is 0 and goes positive and negative on either side
  v_coords = np.arange(-int(np.floor(vert_size/2)), int(np.ceil(vert_size/2)))
  h_coords = np.arange(-int(np.floor(horz_size/2)), int(np.ceil(horz_size/2)))
  return v_coords, h_coords


def _standardize_env_orientation(orientation):
  # our convention is that gabors can have orientation [0, pi]. Because
  # of the symmetry of the gabor envelope, negative orientations have an
  # equivalent positive orientation (-pi/4 gets mapped to 3pi/4, etc.). This
  # function helps us deal with 2pi wrap-around and the reflection from
  # positive to negative orientations. The flipped_orientation flag will help
  # us to apply an equivalent transformation to the fitted phase.
  if np.sign(orientation) == -1:
    if np.abs(orientation % (-2*np.pi)) < np.pi:
      flipped_orientation = True
    else:
      flipped_orientation = False
  else:
    if orientation % (2*np.pi) > np.pi:
      flipped_orientation = True
    else:
      flipped_orientation = False
  return orientation % np.pi, flipped_orientation

def safe_sign(x):
  # because np.sign(0.0) == 0, SMH
  sign_x = np.sign(x)
  if sign_x == 0.0:
    sign_x = 1.0
  return sign_x

def _standardize_phase(phase):
  # We parameterize phase in the interval [-pi, pi]. This just allows us to
  # deal with modulus 2pi and pi ambiguities.
  mod_2pi = phase % (safe_sign(phase)*2*np.pi)
  if abs(mod_2pi) > np.pi:
    # convert to corresponding positive/negative frequency
    return mod_2pi % (-1*safe_sign(mod_2pi)*np.pi)
  else:
    return mod_2pi
