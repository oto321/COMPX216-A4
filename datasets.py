# Simplified version of sklearn.datasets with only the functions needed for the assignment
import numpy as np

def make_circles(n_samples=100, factor=0.8, noise=0.0):
    """Make a large circle containing a smaller circle in 2d.

    A simple toy dataset to visualize clustering and classification
    algorithms.

    Parameters
    ----------
    n_samples : int, dtype=int, default=100
        It is the total number of points generated.
        For odd numbers, the inner circle will have one point more than the
        outer circle.
   
    factor : float, default=.8
        Scale factor between inner and outer circle in the range `[0, 1)`.
 
    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The integer labels (0 or 1) for class membership of each sample.
    """
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    # so as not to have the first point = last point, we set endpoint=False
    linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
    linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)
    outer_circ_x = np.cos(linspace_out)
    outer_circ_y = np.sin(linspace_out)
    inner_circ_x = np.cos(linspace_in) * factor
    inner_circ_y = np.sin(linspace_in) * factor

    X = np.vstack(
        [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]
    ).T
    X += np.random.normal(0.0, noise, X.shape)
    y = np.hstack(
        [np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp)]
    )
  
    return X, y

def make_moons(n_samples=100, noise=0.0):
    """Make two interleaving half circles.

    A simple toy dataset to visualize clustering and classification
    algorithms. 

    Parameters
    ----------
    n_samples : int dtype=int, default=100
        The total number of points generated.
  
    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The integer labels (0 or 1) for class membership of each sample.
    """

   
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack(
        [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]
    ).T
    X += np.random.normal(0.0, noise, X.shape)
    y = np.hstack(
        [np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp)]
    )

    return X, y


