"""The number of worker threads of the engine: ltratio_threads (R/threads.R of the R package dflimit)."""
import os

from . import _engine
from ._utils import dfl_posint


def ltratio_threads(n=None):
    """Number of worker threads of the engine.

    The C engine computes the points of its quadrature rules on a pool of worker threads (POSIX threads, or Windows
    threads on Windows). ``ltratio_threads()`` returns the number of workers; ``ltratio_threads(n)`` sets it.

    When the package is imported the number is taken, in this order, from the environment variable
    ``DFL_NTHREADS``, and otherwise it is the number of processors, capped by the environment variable
    ``OMP_THREAD_LIMIT`` if that is set.

    The engine is called by one Python thread at a time (the calls of other threads wait, without holding the GIL);
    each call uses all the worker threads.

    Parameters
    ----------
    n : int, optional
        ``None`` (the default) to query, or the number of worker threads, a positive integer (1: no threads).

    Returns
    -------
    int
        The number of worker threads in effect.

    Raises
    ------
    TypeError, ValueError
        If ``n`` is not a positive integer.

    Examples
    --------
    >>> from dflimit import ltratio_threads
    >>> old = ltratio_threads()
    >>> ltratio_threads(2)
    2
    >>> ltratio_threads(old) == old
    True
    """
    if n is None:
        return _engine.threads()
    return _engine.threads(dfl_posint(n, "n"))


def default_threads():
    """The number of threads at import: None to leave it to the engine (DFL_NTHREADS, or the number of processors)."""
    if os.environ.get("DFL_NTHREADS"):
        return None                                   # (the engine reads it)
    try:
        omp = int(os.environ.get("OMP_THREAD_LIMIT", ""))
    except ValueError:
        return None
    if omp >= 1:
        return min(omp, _engine.threads())
    return None
