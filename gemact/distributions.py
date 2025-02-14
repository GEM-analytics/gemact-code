from .libraries import *
from . import helperfunctions as hf

quick_setup()
logger = log.name('distributions')


# Distribution
class _Distribution:
    """
    Class representing a probability distribution.
    Python informal private alike class to be inherited.
    """

    @property
    def _dist(self):
        return self._dist

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator (default=None).
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)

        return self._dist.rvs(size=size, random_state=random_state)

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int`` or ``float``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self._dist.cdf(x)

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int`` or ``float``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self._dist.logcdf(x)

    def sf(self, x):
        """
        Survival function, 1 - cumulative distribution function.

        :param x: quantile where the survival function is evaluated.
        :type x: ``int`` or ``float``
        :return: survival function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self._dist.sf(x)

    def logsf(self, x):
        """
        Natural logarithm of the survival function.

        :param x: quantile where the logarithm of the survival function is evaluated.
        :type x: ``int`` or ``float``
        :return: natural logarithm of the survival function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self._dist.logsf(x)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of the cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
        """
        return self._dist.ppf(q=q)

    def isf(self, q):
        """
        Inverse survival function (inverse of sf).

        :param q: level at which the inverse survival function is evaluated.
        :type q: ``float``
        :return: inverse survival function.
        :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
        """
        return self._dist.ppf(1 - q)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’) function.

        :param moments: moments to be returned.
        :type moments: string, optional
        :return: moments.
        :rtype: tuple
        """
        return self._dist.stats(moments=moments)

    def entropy(self):
        """
        (Differential) entropy of the random variable.

        :return: entropy
        :rtype: ``numpy.ndarray``
        """
        return self._dist.entropy()

    def expect(self, func, lb=None, ub=None, conditional=False):
        """
        Expected value of a function (of one argument) with respect to the distribution.

        :param func: function for which integral is calculated. Takes only one argument.
                    The default is the identity mapping f(x) = x.
        :type func: ``callable``, optional
        :param lb: Lower bound for integration. Default is set to the support of the distribution.
        :type lb: ``float``, optional
        :param ub: Upper bound for integration. Default is set to the support of the distribution.
        :type ub: ``float``, optional
        :param conditional: If True, the integral is corrected by the conditional probability of the integration
                        interval.
                        The return value is the expectation of the function, conditional on being in the given interval.
                        Default is False.
        :type conditional: ``bool``, optional
        :return: the calculated expected value.
        :rtype: ``float``

        """
        return self._dist.expect(func, lb=lb, ub=ub, conditional=conditional)

    def median(self):
        """
        Median of the distribution.

        :return: median.
        :rtype: ``float``

        """
        return self._dist.median()

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``float``
        """
        return self._dist.mean()

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``float``

        """
        return self._dist.var()

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``float``

        """
        return self._dist.std()

    def interval(self, alpha):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution.

        :param alpha: fraction alpha
        :type alpha: ``float``
        :return: Endpoints
        :rtype: tuple

        """
        return self._dist.interval(alpha)

    def moment(self, n):
        """
        Non-central moment of order n.

        :param n: moment order.
        :type n: ``int``
        :return: raw moment of order n.
        :rtype: ``float``
        """
        hf.assert_type_value(n, 'n', logger, (float, int), lower_bound=1, lower_close=True)
        n = int(n)
        try:
            output = self._dist.moment(n=n)
        except:
            output = self._dist.moment(order=n)
        return output

    def skewness(self):
        """
        Skewness (third standardized moment).

        :return: skewness.
        :rtype: ``float``
        """
        return self._dist.stats(moments='s').item()
    
    def kurtosis(self):
        """
        Excess kurtosis.

        :return: Excess kurtosis.
        :rtype: ``float``
        """
        return self._dist.stats(moments='k').item()


# Discrete distribution
class _DiscreteDistribution(_Distribution):
    """
    Class representing a discrete probability distribution. To be inherited.
    Child class of ``_Distribution`` class.
    """

    def __init__(self):
        _Distribution.__init__(self)

    @property
    def _dist(self):
        return self._dist

    @staticmethod
    def category():
        return {'frequency'}

    def pmf(self, x):
        """
        Probability mass function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self._dist.pmf(x)

    def logpmf(self, x):
        """
        Natural logarithm of the probability mass function.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``
        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self._dist.logpmf(x)


# Continuous distribution
class _ContinuousDistribution(_Distribution):
    """
    Class representing a continuous probability distribution. To be inherited.
    Child class of ``_Distribution`` class.
    """

    def __init__(self):
        _Distribution.__init__(self)

    @property
    def _dist(self):
        return self._dist

    @staticmethod
    def category():
        return {'severity'}

    def pdf(self, x):
        """
        Probability density function.

        :param x: quantile where probability denisty function is evaluated.
        :type x: ``numpy.ndarray``, ``list``, ``float``, ``int``

        :return: probability density function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self._dist.pdf(x)

    def logpdf(self, x):
        """
        Natural logarithm of the probability denisty function.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``float``
        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self._dist.logpdf(x)

    def fit(self, data):
        """
        Return estimates of shape (if applicable), location, and scale parameters from data.
        The default estimation method is Maximum Likelihood Estimation (MLE),
        but Method of Moments (MM) is also available.
        Refer to ``scipy.stats.rv_continuous.fit``

        :param data: data to use in estimating the distribution parameters.
        :type data: array_like
        :return: parameter_tuple. Estimates for any shape parameters (if applicable),
                followed by those for location and scale.
        :rtype: tuple of floats

        """
        return self._dist.fit(data)
    
    def censored_moment(self, n, d, c):
        """
        Non-central moment of order n of the transformed random variable min(max(x - d, 0), c).
        When n = 1 it is the so-called stop loss transformation function.
        General method for continuous distributions, overridden by distribution specific implementation if available.
        
        :param d: deductible, or attachment point.
        :type d: ``int``, ``float``
        :param c: cover, deductible + cover is the detachment point.
        :type c: ``int``, ``float``
        :param n: moment order.
        :type n: ``int``
        :return: raw moment of order n.
        :rtype: ``float``
        """
        return hf.censored_moment(n=n, d=d, c=c, dist=self)
    
    def partial_moment(self, n, low, up):
        """
        Non-central partial moment of order n.
        General method for continuous distributions, overridden by distribution specific implementation if available.

        :param n: moment order.
        :type n: ``int``
        :param low: lower limit of the partial moment.
        :type low: ``int``, ``float``
        :param up: upper limit of the partial moment.
        :type up: ``int``, ``float``
        :return: raw partial moment of order n.
        :rtype: ``float`` 
        """
        return hf.partial_moment(n=n, low=low, up=up, dist=self) 
    
    def truncated_moment(self, n, low, up):
        """
        Non-central truncated moment of order n.
        General method for continuous distributions, overridden by distribution specific implementation if available.

        :param n: moment order.
        :type n: ``int``
        :param low: lower truncation point.
        :type low: ``int``, ``float``
        :param up: upper truncation point.
        :type up: ``int``, ``float``
        :return: raw truncated moment of order n.
        :rtype: ``float`` 
        """
        adj = self.cdf(up) - self.cdf(low)
        return hf.partial_moment(n=n, low=low, up=up, dist=self) / adj

#Discrete Multivariate Distribution
class _MultDiscreteDistribution(_DiscreteDistribution):
    """
    Class representing a multivariate discrete probability distribution.
    Child class of ``_DiscreteDistribution`` class.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  
        self.marginals = []
        
    def _apply_to_marginals(self, method_name, *args, **kwargs):
        """
        Applies a method of _DiscreteDistribution to the marginal distributions.

        :param method_name: The name of the method of _DiscreteDistribution to be applied.
        :return: Numpy array with the results of applying the method to all marginals.
        """
        results = [getattr(marginal, method_name)(*args, **kwargs) for marginal in self.marginals]
        return np.array(results)
    
    @property
    def _dist(self):
        return self._apply_to_marginals('dist')

    @staticmethod
    def category(self):
        return self._apply_to_marginals('category')

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of the cumulative distribution function of the marginal distributions.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
        """
        return self._apply_to_marginals('ppf', q)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’) function of the marginal distributions.

        :param moments: moments to be returned.
        :type moments: string, optional
        :return: moments.
        :rtype: tuple
        """
        return self._apply_to_marginals('stats', moments)

    def expect(self, func, lb=None, ub=None, conditional=False):
        """
        Expected value of a function (of one argument) with respect to the marginal distributions

        :param func: function for which integral is calculated. Takes only one argument.
                    The default is the identity mapping f(x) = x.
        :type func: ``callable``, optional
        :param lb: Lower bound for integration. Default is set to the support of the distribution.
        :type lb: ``float``, optional
        :param ub: Upper bound for integration. Default is set to the support of the distribution.
        :type ub: ``float``, optional
        :param conditional: If True, the integral is corrected by the conditional probability of the integration
                        interval.
                        The return value is the expectation of the function, conditional on being in the given interval.
                        Default is False.
        :type conditional: ``bool``, optional
        :return: the calculated expected value.
        :rtype: ``float``

        """
        return self._apply_to_marginals('expect', func, lb, ub, conditional)

    def median(self):
        """
        Median of the distribution of the marginal distributions.

        :return: median.
        :rtype: ``float``

        """
        return self._apply_to_marginals('median')

    def mean(self):
        """
        Mean of the distribution of the marginal distributions.

        :return: mean.
        :rtype: ``float``
        """
        return self._apply_to_marginals('mean')

    def var(self):
        """
        Variance of the distribution of the marginal distributions.

        :return: variance.
        :rtype: ``float``

        """
        return self._apply_to_marginals('var')

    def std(self):
        """
        Standard deviation of the distribution of the marginal distributions.

        :return: standard deviation.
        :rtype: ``float``

        """
        return self._apply_to_marginals('std')

    def interval(self, alpha):
        """
        Endpoints of the range that contains fraction alpha [0, 1] of the distribution of the marginal distributions.

        :param alpha: fraction alpha
        :type alpha: ``float``
        :return: Endpoints
        :rtype: tuple

        """
        return self._apply_to_marginals('interval', alpha)

    def moment(self, n):
        """
        Non-central moment of order n of the marginal distributions.

        :param n: moment order.
        :type n: ``int``
        :return: raw moment of order n.
        :rtype: ``float``
        """
        return self._apply_to_marginals('moment', n)

    def skewness(self):
        """
        Skewness (third standardized moment) of the marginal distributions.

        :return: skewness.
        :rtype: ``float``
        """
        return self._apply_to_marginals('skewness')
    
    def kurtosis(self):
        """
        Excess kurtosis of the marginal distributions.

        :return: Excess kurtosis.
        :rtype: ``float``
        """
        return self._apply_to_marginals('kurtosis')


# Poisson
class Poisson(_DiscreteDistribution):
    """
    Poisson distribution.
    Wrapper to scipy poisson distribution (``scipy.stats._discrete_distns.poisson_gen``)
    Refer to :py:class:'~_DiscreteDistribution' for additional details.

    :param loc: location parameter (default=0), to shift the support of the distribution.
    :type loc: ``int``, optional
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *mu* (``numpy.float64``) --
          Poisson distribution parameter mu (rate).

    """

    def __init__(self, loc=0, **kwargs):
        _DiscreteDistribution.__init__(self)
        self.mu = kwargs['mu']
        self.loc = loc

    @property
    def mu(self):
        return self.__mu

    @mu.setter
    def mu(self, value):
        hf.assert_type_value(value, 'mu', logger, (float, int), lower_bound=0, lower_close=False)
        self.__mu = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (float, int))
        value = int(value)
        self.__loc = value

    @property
    def _dist(self):
        return stats.poisson(mu=self.mu, loc=self.loc)

    @property
    def a(self):
        return 0

    @property
    def b(self):
        return self.mu

    @property
    def p0(self):
        return np.exp(-self.mu)

    @staticmethod
    def name():
        return 'poisson'

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, k) parametrization.

        :param f: point where the function is evaluated.
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        return np.exp(self.b * (f - 1))

    def par_deductible_adjuster(self, nu):
        """
        Parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.mu = nu * self.mu
    
    def par_deductible_reverter(self, nu):
        """
        Undo parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.mu = self.mu / nu

    def abk(self):
        """
        Function returning (a, b, k) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, self.p0

    def skewness(self):
        """
        Skewness (third standardized moment).

        :return: skewness.
        :rtype: ``float``
        """
        return 1 / np.sqrt(self.mu)
    
    def kurtosis(self):
        """
        Excess kurtosis.

        :return: Excess kurtosis.
        :rtype: ``float``
        """
        return 1 / self.mu


# Binomial
class Binom(_DiscreteDistribution):
    """
    Binomial distribution.
    Wrapper to scipy binomial distribution (``scipy.stats._discrete_distns.binom_gen``).
    Refer to :py:class:'~_DiscreteDistribution' for additional details.

    :param loc: location parameter (default=0), to shift the support of the distribution.
    :type loc: ``int``, optional
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *n* (``int``) --
          Number of trials.
        * *p* (``float``) --
          Probability of a success, parameter of the binomial distribution.

    """

    def __init__(self, loc=0, **kwargs):
        _DiscreteDistribution.__init__(self)
        self.n = kwargs['n']
        self.p = kwargs['p']
        self.loc = loc

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        hf.assert_type_value(value, 'n', logger, (float, int), lower_bound=1, lower_close=True)
        value = int(value)
        self.__n = value

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        hf.assert_type_value(
            value, 'p', logger, (float, int),
            lower_bound=0, upper_bound=1, lower_close=True, upper_close=True
        )
        self.__p = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (float, int))
        value = int(value)
        self.__loc = value

    @property
    def _dist(self):
        return stats.binom(n=self.n, p=self.p, loc=self.loc)

    @property
    def a(self):
        return -self.p / (1 - self.p)

    @property
    def b(self):
        return (self.n + 1) * (self.p / (1 - self.p))

    @property
    def p0(self):
        return (1 - self.p) ** self.n

    @staticmethod
    def name():
        return 'binom'

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, k) parametrization.

        :param f: point where the function is evaluated
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        return (1 + self.a / (self.a - 1) * (f - 1)) ** (-self.b / self.a - 1)

    def par_deductible_adjuster(self, nu):
        """
        Parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = nu * self.p

    def par_deductible_reverter(self, nu):
        """
        Undo parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = self.p / nu

    def abk(self):
        """
        Function returning (a, b, k) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, self.p0

    def skewness(self):
        """
        Skewness (third standardized moment).

        :return: skewness.
        :rtype: ``float``
        """
        return (1 - 2 * self.p) / np.sqrt(self.n * self.p * (1-self.p))
    
    def kurtosis(self):
        """
        Excess kurtosis.

        :return: Excess kurtosis.
        :rtype: ``float``
        """
        q = 1 - self.p
        return (1 - 6 * self.p * q) / (self.n * self.p * q)


# Geometric
class Geom(_DiscreteDistribution):
    """
    Geometric distribution.
    Wrapper to scipy geometric distribution (``scipy.stats._discrete_distns.geom_gen``).
    Refer to :py:class:'~_DiscreteDistribution' for additional details.

    :param loc: location parameter (default=0), to shift the support of the distribution.
    :type loc: ``int``, optional
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *p* (``float``) --
          Probability parameter of the geometric distribution.

    """

    def __init__(self, loc=0, **kwargs):
        _DiscreteDistribution.__init__(self)
        self.p = kwargs['p']
        self.loc = loc

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        hf.assert_type_value(
            value, 'p', logger, (float, int),
            lower_bound=0, upper_bound=1, lower_close=True, upper_close=True
        )
        self.__p = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (float, int))
        value = int(value)
        self.__loc = value

    @property
    def _dist(self):
        return stats.geom(p=self.p, loc=self.loc)

    @property
    def a(self):
        return 1 - self.p

    @property
    def b(self):
        return 0

    @property
    def p0(self):
        return ((1 - self.p) / self.p) ** (-1)

    @staticmethod
    def name():
        return 'geom'

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, k) parametrization.

        :param f: point where the function is evaluated
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        return (1 - self.a / (1 - self.a) * (f - 1)) ** (-1)

    def par_deductible_adjuster(self, nu):
        """
        Parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = 1 / (1 + nu * (1 - self.p) / self.p)

    def par_deductible_reverter(self, nu):
        """
        Undo parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = 1 / (1 + 1 / nu * (1/self.p - 1))

    def abk(self):
        """
        Function returning (a, b, k) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, self.p0

    def skewness(self):
        """
        Skewness (third standardized moment).

        :return: skewness.
        :rtype: ``float``
        """
        return (2 - self.p) / np.sqrt( 1 - self.p)
    
    def kurtosis(self):
        """
        Excess kurtosis.

        :return: Excess kurtosis.
        :rtype: ``float``
        """
        return 6 + (self.p**2 / (1- self.p))


# Negative Binomial
class NegBinom(_DiscreteDistribution):
    """
    Negative Binomial distribution.
    Wrapper to scipy negative binomial distribution (``scipy.stats._discrete_distns.nbinom_gen``).
    Refer to :py:class:'~_DiscreteDistribution' for additional details.

    :param loc: location parameter (default=0), to shift the support of the distribution.
    :type loc: ``int``, optional
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *n* (``int``) --
          Size parameter of the negative binomial distribution.
        * *p* (``float``) --
          Probability parameter of the negative binomial distribution.

    """

    def __init__(self, loc=0, **kwargs):
        _DiscreteDistribution.__init__(self)
        self.n = kwargs['n']
        self.p = kwargs['p']
        self.loc = loc

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        hf.assert_type_value(value, 'n', logger, (float, int), lower_bound=1, lower_close=True)
        value = int(value)
        self.__n = value

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        hf.assert_type_value(
            value, 'p', logger, (float, int),
            lower_bound=0, upper_bound=1, lower_close=True, upper_close=True
        )
        self.__p = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (float, int))
        value = int(value)
        self.__loc = value

    @property
    def _dist(self):
        return stats.nbinom(n=self.n, p=self.p, loc=self.loc)

    @property
    def a(self):
        return 1 - self.p

    @property
    def b(self):
        return (self.n - 1) * (1 - self.p)

    @property
    def p0(self):
        return self.p ** self.n

    @staticmethod
    def name():
        return 'nbinom'

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, k) parametrization.

        :param f: point where the function is evaluated.
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        return (1 - self.a / (1 - self.a) * (f - 1)) ** (-self.b / self.a - 1)

    def par_deductible_adjuster(self, nu):
        """
        Parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = 1 / (1 + nu * (1 - self.p) / self.p)

    def par_deductible_reverter(self, nu):
        """
        Undo parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = 1 / (1 + 1 / nu * (1/self.p - 1))

    def abk(self):
        """
        Function returning (a, b, k) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, self.p0

    def skewness(self):
        """
        Skewness (third standardized moment).

        :return: skewness.
        :rtype: ``float``
        """
        return (2 - self.p) / np.sqrt((1 - self.p)*self.n)
    
    def kurtosis(self):
        """
        Excess kurtosis.

        :return: Excess kurtosis.
        :rtype: ``float``
        """
        return 6/self.n + self.p**2 / ((1- self.p)*self.n)


# Logser
class Logser(_DiscreteDistribution):
    """
    Logarithmic (Log-Series, Series) discrete distribution.
    Wrapper to scipy logser distribution (``scipy.stats._discrete_distns.logser_gen``)
    Refer to :py:class:'~_DiscreteDistribution' for additional details.

    :param loc: location parameter (default=0), to shift the support of the distribution.
    :type loc: ``int``, optional
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *p* (``float``) --
          Probability parameter of the logser distribution.

    """

    def __init__(self, loc=0, **kwargs):
        _DiscreteDistribution.__init__(self)
        self.p = kwargs['p']
        self.loc = loc

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        hf.assert_type_value(
            value, 'p', logger, (float, int),
            lower_bound=0, upper_bound=1, lower_close=True, upper_close=True
        )
        self.__p = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (float, int))
        value = int(value)
        self.__loc = value

    @property
    def a(self):
        return self.p
    
    @property
    def b(self):
        return -self.p

    @property
    def p0(self):
        return 0

    @property
    def _dist(self):
        return stats.logser(p=self.p, loc=self.loc)

    @staticmethod
    def category():
        return {'frequency', 'zt'}

    @staticmethod
    def name():
        return 'logser'

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, k) parametrization.

        :param f: point where the function is evaluated.
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        if not np.abs(f) < 1 / self.p:
            logger.error('Make sure f is lower than or equal to %r.' % self.p)
            raise ValueError
        return np.log(1 - self.p * f) / np.log(1 - self.p)
    
    def abk(self):
        """
        Function returning (a, b, k) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, self.p0
    
    def par_deductible_adjuster(self, nu):
        """
        Parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = nu * self.p / (1 - self.p + self.p * nu)
    
    def par_deductible_reverter(self, nu):
        """
        Undo parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = self.p / (nu - self.p * nu + self.p)


# Zero-truncated Poisson
class ZTPoisson:
    """
    Zero-truncated Poisson distribution.
    Poisson distribution with no mass (truncated) in 0.
    scipy reference non-zero-truncated distribution: ``scipy.stats._discrete_distns.poisson_gen``

    :param loc: location parameter (default=0), to shift the support of the distribution.
    :type loc: ``int``, optional
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *mu* (``numpy.float64``) --
          Zero-truncated Poisson distribution parameter mu (rate).

    """

    def __init__(self, loc=0, **kwargs):
        self.mu = kwargs['mu']
        self.loc = loc

    @property
    def mu(self):
        return self.__mu

    @mu.setter
    def mu(self, value):
        hf.assert_type_value(value, 'mu', logger, (float, int), lower_bound=0, lower_close=False)
        self.__mu = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (float, int))
        value = int(value)
        self.__loc = value

    @property
    def _dist(self):
        return stats.poisson(mu=self.mu, loc=self.loc)

    @property
    def a(self):
        return 0

    @property
    def b(self):
        return self.mu

    @property
    def p0(self):
        # probability in 0 of the non-truncated poisson
        return np.exp(-self.mu)

    @staticmethod
    def category():
        return {'frequency', 'zt'}

    @staticmethod
    def name():
        return 'ztpoisson'

    def pmf(self, x):
        """
        Probability mass function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        temp = self._dist.pmf(x) / (1 - self.p0)
        x = np.array(x)
        zeros = np.where(x == 0)[0]
        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = 0.0
            else:
                temp[zeros] = 0.0
            return temp

    def logpmf(self, x):
        """
        Natural logarithm of the probability mass function.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``

        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        return np.log(self.pmf(x))

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``float``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """

        return (self._dist.cdf(x) - self._dist.cdf(0)) / (1 - self._dist.cdf(0))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return np.log(self.cdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)

        q_ = np.random.uniform(low=self._dist.cdf(0), high=1, size=size)
        return self._dist.ppf(q_)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """

        q_ = np.array(q)
        return self._dist.ppf(q=q_ * (1 - self._dist.cdf(0)) + self._dist.cdf(0))

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return self.mu / (1 - np.exp(-self.mu))

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return self.mean() * (1 + self.mu - self.mean())

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, k) parametrization.

        :param f: point where the function is evaluated
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        return (np.exp(self.b * f) - 1) / (np.exp(self.b) - 1)

    def abk(self):
        """
        Function returning (a, b, k) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, 0

    def par_deductible_adjuster(self, nu):
        """
        Parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.mu = nu * self.mu
    
    def par_deductible_reverter(self, nu):
        """
        Undo parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.mu = self.mu / nu


# Zero-modified Poisson
class ZMPoisson:
    """
    Zero-modified Poisson distribution. Discrete mixture between a degenerate distribution
    at zero and a non-modified Poisson distribution.
    scipy reference non-zero-modified distribution: ``scipy.stats._discrete_distns.poisson_gen``

    :param loc: location parameter (default=0).
    :type loc: ``int``, optional
    :param maxdiff: threshold to determine which method to generate random variates (default=0.95).
    :type maxdiff: ``float``, optional
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *mu* (``numpy.float64``) --
          Zero-modified Poisson distribution rate parameter.
        * *p0m* (``numpy.float64``) --
          Zero-modified Poisson mixing parameter. Resulting probability mass in zero.

    """

    def __init__(self, loc=0, maxdiff=0.95, **kwargs):
        self.loc = loc
        self.maxdiff = maxdiff
        self.mu = kwargs['mu']
        self.p0m = kwargs['p0m']

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (int, float))
        value = int(value)
        self.__loc = value

    @property
    def mu(self):
        return self.__mu

    @mu.setter
    def mu(self, value):
        hf.assert_type_value(value, 'mu', logger, (float, int), lower_bound=0, lower_close=False)
        self.__mu = value

    @property
    def p0m(self):
        return self.__p0m

    @p0m.setter
    def p0m(self, value):
        hf.assert_type_value(
            value, 'p0m', logger, (float, int),
            lower_bound=0, upper_bound=1, lower_close=True, upper_close=True
        )
        self.__p0m = value

    @property
    def maxdiff(self):
        return self.__maxdiff

    @maxdiff.setter
    def maxdiff(self, value):
        self.__maxdiff = value

    @property
    def _dist(self):
        return stats.poisson(mu=self.mu, loc=self.loc)

    @property
    def a(self):
        return 0

    @property
    def b(self):
        return self.mu

    @property
    def p0(self):
        return np.exp(-self.mu)

    @staticmethod
    def category():
        return {'frequency', 'zm'}

    @staticmethod
    def name():
        return 'zmpoisson'

    def pmf(self, x):
        """
        Probability mass function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        temp = self._dist.pmf(x) * (1 - self.p0m) / (1 - self.p0)
        x = np.array(x)
        zeros = np.where(x == 0)[0]
        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = self.p0m
            else:
                temp[zeros] = self.p0m
            return temp

    def logpmf(self, x):
        """
        Natural logarithm of the probability mass function.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``

        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        return np.log(self.pmf(x))

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.p0m + (1 - self.p0m) * (self._dist.cdf(x) - self._dist.cdf(0)) / (1 - self._dist.cdf(0))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.cdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)
        
        if self.mu == 0:
            u_ = np.random.uniform(0, 1, size)
            idx = u_ <= self.p0m
            u_[idx] = 0
            u_[np.invert(idx)] = 1
            return u_

        if self.p0m >= self.p0:
            u_ = np.random.uniform(0, (1 - self.p0), size)
            idx = (u_ <= (1 - self.p0m))
            u_[idx] = self._dist.rvs(size=np.sum(idx))
            u_[np.invert(idx)] = 0
            return u_

        if (self.p0 - self.p0m) < self.maxdiff:
            # rejection method
            u_ = []
            while len(u_) < size:
                x_ = self._dist.rvs(1, self.mu)
                if (x_ != 0 or np.random.uniform(0, self.p0 * (1 - self.p0m), 1)) <= ((1 - self.p0) * self.p0m):
                    u_.append(x_)
            return np.asarray(u_)
        else:
            # inversion method
            u_ = np.random.uniform((self.p0 - self.p0m) / (1 - self.p0m), 1, size)
            return self._dist.ppf(u_, self.mu)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cumultaive distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """

        q_ = np.array(q)
        temp = self._dist.ppf((1 - self._dist.cdf(0)) * (q_ - self.p0m) / (1 - self.p0m) + self._dist.cdf(0))
        return temp

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, k) parametrization.

        :param f: point where the function is evaluated
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        return self.p0m + (1 - self.p0m) * (np.exp(self.b * f) - 1) / (np.exp(self.b) - 1)

    def abk(self):
        """
        Function returning (a, b, k) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, self.p0m

    def par_deductible_adjuster(self, nu):
        """
        Parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p0m = (self.p0m - np.exp(-self.mu) + np.exp(-nu * self.mu) - self.p0m * np.exp(-nu * self.mu)) / (
                1 - np.exp(-self.mu))
        self.mu = nu * self.mu
    
    def par_deductible_reverter(self, nu):
        """
        Undo parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.mu = self.mu / nu
        self.p0m = (self.p0m * (1- np.exp(-self.mu)) + np.exp(-self.mu) - np.exp(-nu * self.mu)) / (
                1 - np.exp(-nu * self.mu))

  
# Zero-truncated binomial
class ZTBinom:
    """
    Zero-truncated binomial distribution. Binomial distribution with no mass (truncated) in 0.
    scipy reference non-zero-truncated distribution: ``scipy.stats._discrete_distns.binom_gen`` .

    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *n* (``int``) --
          Zero-truncated binomial distribution size parameter n.
        * *p* (``float``) --
          Zero-truncated binomial distribution probability parameter p.
    """

    def __init__(self, **kwargs):
        self.n = kwargs['n']
        self.p = kwargs['p']

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        hf.assert_type_value(value, 'n', logger, (float, int), lower_bound=1, lower_close=True)
        value = int(value)
        self.__n = value

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        hf.assert_type_value(
            value, 'p', logger, (float, int),
            lower_bound=0, upper_bound=1, lower_close=True, upper_close=True
        )
        self.__p = value

    @property
    def _dist(self):
        return stats.binom(n=self.n, p=self.p)

    @property
    def a(self):
        return -self.p / (1 - self.p)

    @property
    def b(self):
        return (self.n + 1) * (self.p / (1 - self.p))

    @property
    def p0(self):
        # probability in 0 of the non-truncated binom
        return (1 - self.p) ** self.n

    @staticmethod
    def category():
        return {'frequency', 'zt'}

    @staticmethod
    def name():
        return 'ztbinom'

    def pmf(self, x):
        """
        Probability mass function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """

        temp = self._dist.pmf(x) / (1 - self.p0)
        x = np.array(x)
        zeros = np.where(x == 0)[0]
        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = 0.
            else:
                temp[zeros] = 0.0
            return temp

    def logpmf(self, x):
        """
        Natural logarithm of the probability mass function.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``

        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        return np.log(self.pmf(x))

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return (self._dist.cdf(x) - self._dist.cdf(0)) / (1 - self._dist.cdf(0))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return np.log(self.cdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)

        q_ = np.random.uniform(low=self._dist.cdf(0), high=1, size=size)
        return self._dist.ppf(q_)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """

        q_ = np.array(q)
        return self._dist.ppf(q=q_ * (1 - self._dist.cdf(0)) + self._dist.cdf(0))

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return self.n * self.p / (1 - (1 - self.p)**self.n)

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        np_ = self.n * self.p
        q_ = 1 - self.p
        return np_ * (q_ - (q_ + np_) * q_**self.n) / (1 - q_**self.n)**2

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, k) parametrization.

        :param f: point where the function is evaluated
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        a_ = self.a
        b_ = self.b
        return ((1 + a_ / (a_ - 1) * (f - 1)) ** (-b_ / a_ - 1) - (1 - a_) ** (b_ / a_ + 1)) / (
                1 - (1 - a_) ** (b_ / a_ + 1))

    def abk(self):
        """
        Function returning (a, b, k) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, 0

    def par_deductible_adjuster(self, nu):
        """
        Parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = nu * self.p

    def par_deductible_reverter(self, nu):
        """
        Undo parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = self.p / nu


# Zero-modified binomial
class ZMBinom:
    """
    Zero-modified binomial distribution. Discrete mixture between a degenerate distribution
    at zero and a non-modified binomial distribution.
    scipy reference non-zero-modified distribution: ``scipy.stats._discrete_distns.binom_gen``.

    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *n* (``numpy.float64``) --
          Zero-modified binomial distribution size parameter n.
        * *p* (``numpy.float64``) --
          Zero-modified binomial distribution probability parameter p.
        * *p0m* (``numpy.float64``) --
          Zero-modified binomial mixing parameter.

    """

    def __init__(self, **kwargs):
        self.n = kwargs['n']
        self.p = kwargs['p']
        self.p0m = kwargs['p0m']

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        hf.assert_type_value(value, 'n', logger, (float, int), lower_bound=1)
        value = int(value)
        self.__n = value

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        hf.assert_type_value(
            value, 'p', logger, (float, int),
            lower_bound=0, upper_bound=1, lower_close=True, upper_close=True
        )
        self.__p = value

    @property
    def p0m(self):
        return self.__p0m

    @p0m.setter
    def p0m(self, value):
        hf.assert_type_value(
            value, 'p0m', logger, (float, int),
            lower_bound=0, upper_bound=1, lower_close=True, upper_close=True
        )
        self.__p0m = value

    @property
    def _dist(self):
        return stats.binom(n=self.n, p=self.p)

    @property
    def _distzt(self):
        return ZTBinom(n=self.n, p=self.p)

    @property
    def a(self):
        return -self.p / (1 - self.p)

    @property
    def b(self):
        return (self.n + 1) * (self.p / (1 - self.p))

    @property
    def p0(self):
        return (1 - self.p) ** self.n

    @staticmethod
    def category():
        return {'frequency', 'zm'}

    @staticmethod
    def name():
        return 'zmbinom'

    def pmf(self, x):
        """
        Probability mass function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self._dist.pmf(x) * (1 - self.p0m) / (1 - self.p0)

    def logpmf(self, x):
        """
        Natural logarithm of the probability mass function.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``

        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.pmf(x))

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.p0m + (1 - self.p0m) * (self._dist.cdf(x) - self._dist.cdf(0)) / (1 - self._dist.cdf(0))

    def logcdf(self, x):
        """
        Natural lograithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.cdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)
        
        r_ = stats.bernoulli(p=1 - self.p0m).rvs(size, random_state=random_state)
        c_ = np.where(r_ == 1)[0]
        if int(len(c_)) > 0:
            r_[c_] = self._distzt.rvs(int(len(c_)))
        return r_

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        q_ = np.array(q)
        temp = self._dist.ppf((1 - self._dist.cdf(0)) * (q_ - self.p0m) / (1 - self.p0m) + self._dist.cdf(0))
        return temp

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, k) parametrization.

        :param f: point where the function is evaluated
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        a_ = self.a
        b_ = self.b
        return self.p0m + (1 - self.p0m) * ((1 + a_ / (a_ - 1) * (f - 1)) ** (-b_ / a_ - 1) - (
                1 - a_) ** (b_ / a_ + 1)) / (1 - (1 - a_) ** (b_ / a_ + 1))

    def abk(self):
        """
        Function returning (a, b, k) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, self.p0m

    def par_deductible_adjuster(self, nu):
        """
        Parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p0m = (self.p0m - (1 - self.p) ** self.n + (
                1 - nu * self.p) ** self.n - self.p0m * (1 - nu * self.p) ** self.n) / (
                           1 - (1 - self.p) ** self.n)
        self.p = nu * self.p
    
    def par_deductible_reverter(self, nu):
        """
        Undo parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = self.p / nu
        self.p0m = (self.p0m + (1 - self.p)**self.n - (1 - nu * self.p)**self.n) / (
                    1 - (1 -  nu * self.p) ** self.n)


# Zero-truncated geometric
class ZTGeom:
    """
    Zero-truncated geometric distribution. Geometric distribution with no mass (truncated) in 0.
    scipy reference non-zero-truncated distribution: ``scipy.stats._discrete_distns.geom_gen``

    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *p* (``numpy.float64``) --
          Zero-truncated geometric distribution probability parameter p.

    """

    def __init__(self, **kwargs):
        self.p = kwargs['p']

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        hf.assert_type_value(
            value, 'p', logger, (float, int),
            lower_bound=0, upper_bound=1, lower_close=True, upper_close=True
        )
        self.__p = value

    @property
    def _dist(self):
        return stats.geom(p=self.p)

    @property
    def a(self):
        return 1 - self.p

    @property
    def b(self):
        return 0

    @property
    def p0(self):
        return self.p

    @staticmethod
    def category():
        return {'frequency', 'zt'}

    @staticmethod
    def name():
        return 'ztgeom'

    def pmf(self, x):
        """
        Probability mass function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        temp = (self._dist.pmf(x + 1)) / (1 - self.p0)
        x = np.array(x)
        zeros = np.where(x == 0)[0]
        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = 0.
            else:
                temp[zeros] = 0.0
            return temp

    def logpmf(self, x):
        """
        Natural logarithm of the probability mass function.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``

        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        return np.log(self.pmf(x))

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return (self._dist.cdf(x) - self._dist.cdf(0)) / (1 - self._dist.cdf(0))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.cdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)

        q_ = np.random.uniform(low=self._dist.cdf(0), high=1, size=size)
        return self._dist.ppf(q_)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """

        q_ = np.array(q)
        return self._dist.ppf(q=q_ * (1 - self._dist.cdf(0)) + self._dist.cdf(0))

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return 1 / self.p

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return (1 - self.p) / self.p**2
    
    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, k) parametrization.

        :param f: point where the function is evaluated
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        return (1 / (1 - (f - 1) / (1 - self.a)) - 1 + self.a) / self.a

    def abk(self):
        """
        Function returning (a, b, k) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, 0

    def par_deductible_adjuster(self, nu):
        """
        Parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = 1 / (1 + nu * (1 - self.p) / self.p)

    def par_deductible_reverter(self, nu):
        """
        Undo parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = 1 / ((1 / self.p - 1) / nu + 1)


# Zero-modified geometric
class ZMGeom:
    """
    Zero-modified geometric distribution. Discrete mixture between a degenerate distribution
    at zero and a non-modified geometric distribution.
    scipy reference non-zero-modified distribution: `scipy.stats._discrete_distns.geom_gen``

    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *p* (``numpy.float64``) --
          Zero-modified geometric distribution probability parameter p.
        * *p0m* (``numpy.float64``) --
          Zero-modified geometric mixing parameter.
    """

    def __init__(self, **kwargs):
        self.p = kwargs['p']
        self.p0m = kwargs['p0m']

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        hf.assert_type_value(
            value, 'p', logger, (float, int),
            lower_bound=0, upper_bound=1, lower_close=True, upper_close=True
        )
        self.__p = value

    @property
    def p0m(self):
        return self.__p0m

    @p0m.setter
    def p0m(self, value):
        hf.assert_type_value(
            value, 'p0m', logger, (float, int),
            lower_bound=0, upper_bound=1, lower_close=True, upper_close=True
        )
        self.__p0m = value

    @property
    def a(self):
        return 1 - self.p

    @property
    def b(self):
        return 0

    @property
    def p0(self):
        return self.p

    @property
    def _dist(self):
        return stats.geom(p=self.p)

    @property
    def _distzt(self):
        return ZTGeom(p=self.p)

    @staticmethod
    def category():
        return {'frequency', 'zm'}

    @staticmethod
    def name():
        return 'zmgeom'

    def pmf(self, x):
        """
        Probability mass function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return (self._dist.pmf(x) + 1) * (1 - self.p0m) / (1 - self.p0)

    def logpmf(self, x):
        """
        Natural logarithm of the probability mass function.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``

        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.pmf(x))

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.p0m + (1 - self.p0m) * (self._dist.cdf(x) - self._dist.cdf(0)) / (1 - self._dist.cdf(0))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.cdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """

        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)
        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)

        r_ = stats.bernoulli(p=1 - self.p0m).rvs(size, random_state=random_state)
        c_ = np.where(r_ == 1)[0]
        if int(len(c_)) > 0:
            r_[c_] = self._distzt.rvs(int(len(c_)))
        return r_

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        q_ = np.array(q)
        return self._dist.ppf((1 - self._dist.cdf(0)) * (q_ - self.p0m) / (1 - self.p0m) + self._dist.cdf(0))

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, k) parametrization.

        :param f: point where the function is evaluated
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        return self.p0m + (1 - self.p0m) * (1 / (1 - (f - 1) / (1 - self.a)) - 1 + self.a) / self.a

    def abk(self):
        """
        Function returning (a, b, k) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, self.p0m

    def par_deductible_adjuster(self, nu):
        """
        Parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        beta = (1 - self.p) / self.p
        self.p0m = (self.p0m - (1 + beta) ** -1 + (1 + nu * beta) ** -1 - self.p0m * (1 + nu * beta) ** -1) / (
                1 - (1 + beta) ** -1)
        self.p = 1 / (1 + nu * beta)

    def par_deductible_reverter(self, nu):
        """
        Undo parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        beta = (1 - self.p) / self.p
        self.p = nu / (nu + beta)
        self.p0m = ((1 - (1 + beta) ** -1) * self.p0m + (1 + beta) ** -1 - (1 + nu * beta) ** -1) / (1 - (1 + nu * beta) ** -1)


# Zero-truncated negative binomial
class ZTNegBinom:
    """
    Zero-truncated negative binomial distribution. Negative binomial distribution with no mass (truncated) in 0.
    scipy reference non-zero-truncated distribution: ``scipy.stats._discrete_distns.nbinom_gen``.

    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *n* (``int``) --
          Zero-truncated negative binomial distribution size parameter n.
        * *p* (``numpy.float64``) --
          Zero-truncated negative binomial distribution probability parameter p.
    """

    def __init__(self, **kwargs):
        self.n = kwargs['n']
        self.p = kwargs['p']

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        hf.assert_type_value(value, 'n', logger, (float, int), lower_bound=0)
        value = int(value)
        self.__n = value

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        hf.assert_type_value(
            value, 'p', logger, (float, int),
            lower_bound=0, upper_bound=1, lower_close=True, upper_close=True
        )
        self.__p = value

    @property
    def a(self):
        return 1 - self.p

    @property
    def b(self):
        return (self.n - 1) * (1 - self.p)

    @property
    def p0(self):
        return self.p ** self.n

    @property
    def _dist(self):
        return stats.nbinom(n=self.n, p=self.p)

    @staticmethod
    def category():
        return {'frequency', 'zt'}

    @staticmethod
    def name():
        return 'ztnbinom'

    def pmf(self, x):
        """
        Probability mass function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        temp = (self._dist.pmf(x)) / (1 - self.p0)
        x = np.array(x)
        zeros = np.where(x == 0)[0]
        if zeros.size == 0:
            return temp
        else:
            if x.shape == ():
                temp = 0.
            else:
                temp[zeros] = 0.0
            return temp

    def logpmf(self, x):
        """
        Natural logarithm of the probability mass function.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``

        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.pmf(x))

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return (self._dist.cdf(x) - self._dist.cdf(0)) / (1 - self._dist.cdf(0))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.cdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)

        q_ = np.random.uniform(low=self._dist.cdf(0), high=1, size=size)
        return self._dist.ppf(q_)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        q_ = (np.array(q) * (1 - self._dist.cdf(0)) + self._dist.cdf(0))
        return self._dist.ppf(q=q_)

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return self.n * (1 - self.p) / self.p * (1 - self.p**self.n)

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return (self.n * (1 - self.p) * (1 - (1 + self.n * (1 - self.p)) * self.p**self.n)) / (self.p * (1 - self.p**self.n))**2
    
    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, k) parametrization.

        :param f: point where the function is evaluated
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        c_ = self.b / self.a + 1
        d_ = 1 - self.a
        return ((1 / (1 - (f - 1) * self.a / d_)) ** c_ - d_ ** c_) / (1 - d_ ** c_)

    def abk(self):
        """
        Function returning (a, b, k) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, 0

    def par_deductible_adjuster(self, nu):
        """
        Parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = 1 / (1 + nu * (1 - self.p) / self.p)

    def par_deductible_reverter(self, nu):
        """
        Undo parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = (1 + nu) / ((1 / self.p) + nu)


# Zero-modified negative binomial
class ZMNegBinom:
    """
    Zero-modified negative binomial distribution. Discrete mixture between a degenerate distribution
    at zero and a non-modified negative binomial distribution.
    scipy reference non-zero-modified distribution: ``scipy.stats._discrete_distns.nbinom_gen``.

    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *n* (``int``) --
          Zero-modified negative binomial distribution size parameter n.
        * *p* (``numpy.float64``) --
          Zero-modified negative binomial distribution probability parameter p.
        * *p0m* (``numpy.float64``) --
          Zero-modified negative binomial mixing parameter.

    """

    def __init__(self, **kwargs):
        self.n = kwargs['n']
        self.p = kwargs['p']
        self.p0m = kwargs['p0m']

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        hf.assert_type_value(value, 'n', logger, (float, int), lower_bound=0)
        value = int(value)
        self.__n = value

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        hf.assert_type_value(
            value, 'p', logger, (float, int),
            lower_bound=0, upper_bound=1, lower_close=True, upper_close=True
        )
        self.__p = value

    @property
    def p0m(self):
        return self.__p0m

    @p0m.setter
    def p0m(self, value):
        hf.assert_type_value(
            value, 'p0m', logger, (float, int),
            lower_bound=0, upper_bound=1, lower_close=True, upper_close=True
        )
        self.__p0m = value

    @property
    def a(self):
        return 1 - self.p

    @property
    def b(self):
        return (self.n - 1) * (1 - self.p)

    @property
    def p0(self):
        return (1 / self.p) ** -self.n

    @property
    def _dist(self):
        return stats.nbinom(n=self.n, p=self.p)

    @property
    def _distzt(self):
        return ZTNegBinom(n=self.n, p=self.p)

    @staticmethod
    def category():
        return {'frequency', 'zm'}

    @staticmethod
    def name():
        return 'zmnbinom'

    def pmf(self, x):
        """
        Probability mass function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return (self._dist.pmf(x) * (1 - self.p0m)) / (1 - self.p0)

    def logpmf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        return np.log(self.pmf(x))

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: int
        :return: cumulative distribution function.
        :rtype: numpy.float64 or numpy.ndarray

        """
        return self.p0m + (1 - self.p0m) * (self._dist.cdf(x) - self._dist.cdf(0)) / (1 - self._dist.cdf(0))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.cdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)

        r_ = stats.bernoulli(p=1 - self.p0m).rvs(size, random_state=random_state)
        c_ = np.where(r_ == 1)[0]
        if int(len(c_)) > 0:
            r_[c_] = self._distzt.rvs(int(len(c_)))
        return r_

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """

        p_ = np.array(q)
        return self._dist.ppf((1 - self._dist.cdf(0)) * (p_ - self.p0m) / (1 - self.p0m) + self._dist.cdf(0))

    def pgf(self, f):
        """
        Probability generating function. It computes the probability generating function
        of the random variable given the (a, b, k) parametrization.

        :param f: point where the function is evaluated
        :type f: ``numpy array``
        :return: probability generated in f.
        :rtype: ``numpy.ndarray``
        """
        c_ = 1 - self.a
        d_ = (self.b / self.a + 1)

        return self.p0m + (1 - self.p0m) * ((1 / (1 - (f - 1) * self.a / c_)) ** d_ - c_ ** d_) / (1 - c_ ** d_)

    def abk(self):
        """
        Function returning (a, b, k) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, self.p0m

    def par_deductible_adjuster(self, nu):
        """
        Parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        beta = (1 - self.p) / self.p
        self.p0m = (self.p0m - (1 + beta) ** (-self.n) + (1 + nu * beta) ** -self.n - self.p0m * (
                1 + nu * beta) ** -self.n) / (1 - (1 + beta) ** -self.n)
        self.p = 1 / (1 + nu * beta)

    def par_deductible_reverter(self, nu):
        """
        Undo parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        beta = (1 - self.p) / self.p
        self.p = nu / (nu + beta)
        self.p0m = (self.p0m * (1 - (1 + beta) ** -self.n) +  (1 + beta) ** -self.n - ((1 + beta) ** -self.n)) / (
            1 - (1 + nu * beta) ** -self.n)


# Zero-modified discrete logarithmic
class ZMLogser:
    """
    Zero-modified (discrete) logarithmic (log-series) distribution.
    Discrete mixture between a degenerate distribution
    at zero and a non-modified logarithmic distribution.
    scipy reference non-zero-modified distribution: ``scipy.stats._discrete_distns.logser_gen``

    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *p* (``numpy.float64``) --
          ZM discrete logarithmic distribution probability parameter p.
        * *p0m* (``numpy.float64``) --
          ZM discrete logarithmic mixing parameter.
    """

    def __init__(self, **kwargs):
        self.p = kwargs['p']
        self.p0m = kwargs['p0m']

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        hf.assert_type_value(
            value, 'p', logger, (float, int),
            lower_bound=0, upper_bound=1, lower_close=True, upper_close=True
        )
        self.__p = value

    @property
    def p0m(self):
        return self.__p0m

    @p0m.setter
    def p0m(self, value):
        hf.assert_type_value(
            value, 'p0m', logger, (float, int),
            lower_bound=0, upper_bound=1, lower_close=True, upper_close=True
        )
        self.__p0m = value

    @property
    def p0(self):
        # probability in 0 of the un-modified logser distribution.
        return 0

    @property
    def a(self):
        return self.p
    
    @property
    def b(self):
        return -self.p

    @property
    def _dist(self):
        return stats.logser(p=self.p)

    @staticmethod
    def category():
        return {'frequency', 'zm'}

    @staticmethod
    def name():
        return 'zmlogser'

    def pmf(self, x):
        """
        Probability mass function.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return (self._dist.pmf(x) * (1 - self.p0m)) / (1 - self.p0)

    def logpmf(self, x):
        """
        Natural logarithm of the probability mass function.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``

        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        return np.log(self.pmf(x))

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self.p0m + (1 - self.p0m) * (self._dist.cdf(x) - self._dist.cdf(0)) / (1 - self._dist.cdf(0))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: quantile where log of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.cdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)

        q_ = np.random.uniform(0, 1, size)
        return self.ppf(q_)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """

        p_ = np.array(q)
        temp = self._dist.ppf((1 - self._dist.cdf(0)) * (p_ - self.p0m) / (1 - self.p0m) + self._dist.cdf(0))

        zeros = np.where(p_ <= self.p0m)[0]
        if zeros.size == 0:
            return temp
        else:
            if p_.shape == ():
                temp = self.p0m
            else:
                temp[zeros] = self.p0m
            return temp

    def abk(self):
        """
        Function returning (a, b, k) parametrization.

        :return: a, b, probability in zero
        :rtype: ``numpy.array``
        """
        return self.a, self.b, self.p0m

    def par_deductible_adjuster(self, nu):
        """
        Parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p0m = 1 + (1 - self.p0m) * np.log(1 + nu * self.p / (1 - self.p)) / np.log(1 - self.p)
        self.p = nu * self.p / (1 - self.p + self.p * nu)

    def par_deductible_reverter(self, nu):
        """
        Undo parameter correction in case of deductible.

        :param nu: severity model survival function at the deductible.
        :type nu: ``float``
        :return: Void
        :rtype: None
        """
        self.p = self.p / (nu - self.p * nu + self.p)
        self.p0m = 1 - (self.p0m - 1) * np.log(1 - self.p) / np.log(1 + nu * self.p / (1 - self.p))


# Beta
class Beta(_ContinuousDistribution):
    """
    Wrapper to scipy beta distribution.
    ``scipy.stats._continuous_distns.beta_gen``

    :param scale: beta scale parameter.
    :type scale: ``float``
    :param loc: beta location parameter.
    :type loc: ``float``
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *a* (``int`` or ``float``) --
          shape parameter a.
        * *b* (``int`` or ``float``) --
          shape parameter b.
    """

    def __init__(self, loc=0, scale=1, **kwargs):
        _ContinuousDistribution.__init__(self)
        self.a = kwargs['a']
        self.b = kwargs['b']
        self.scale = scale
        self.loc = loc

    @property
    def a(self):
        return self.__a

    @a.setter
    def a(self, value):
        hf.assert_type_value(value, 'a', logger, (float, int), lower_bound=0, lower_close=False)
        self.__a = value

    @property
    def b(self):
        return self.__b

    @b.setter
    def b(self, value):
        hf.assert_type_value(value, 'b', logger, (float, int), lower_bound=0, lower_close=False)
        self.__b = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (float, int))
        self.__loc = value

    @property
    def _dist(self):
        return stats.beta(
            a=self.a,
            b=self.b,
            loc=self.loc,
            scale=self.scale
        )

    @staticmethod
    def name():
        return 'beta'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        hf.assert_type_value(v, 'v', logger, (np.floating, np.integer, np.ndarray, int, float))
        v = np.array([v]).flatten()
        output = v.copy().astype(np.float64)
        u = v / self.scale
        output[v == np.inf] = self.mean()
        flt = (v > 0) * (v < np.inf)
        output[flt] = v[flt] * (
                1 - special.betaincinv(self.a, self.b, u[flt])) + special.betaincinv(
            self.a + 1, self.b, u[flt]) * self.scale * special.gamma(
            self.a + self.b) * special.gamma(
            self.a + 1) / (special.gamma(self.a + self.b + 1) * special.gamma(self.a))
        return output


# Exponential
class Exponential(_ContinuousDistribution):
    """
    Expontential distribution.
    scipy reference distribution: ``scipy.stats._continuous_distns.expon_gen``

    :param theta: exponential distribution theta parameter.
    :type theta: ``float``
    :param loc: location parameter
    :type loc: ``float``
    """

    def __init__(self, loc=0, theta=1):
        _ContinuousDistribution.__init__(self)
        self.theta = theta
        self.loc = loc

    @property
    def theta(self):
        return self.__theta

    @theta.setter
    def theta(self, value):
        hf.assert_type_value(
            value, 'theta', logger, (float, int),
            lower_bound=0, lower_close=False
            )
        self.__theta = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (float, int))
        self.__loc = value

    @property
    def _dist(self):
        return stats.expon(loc=self.loc)

    @staticmethod
    def name():
        return 'exponential'

    def pdf(self, x):
        """
        Probability density function.

        :param x: quantile where probability density function is evaluated.
        :type x: ``numpy.ndarray``, ``list``, ``float``, ``int``

        :return: probability density function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self.theta * self._dist.pdf(self.loc + self.theta * x)

    def logpdf(self, x):
        """
        Natural logarithm of the probability density function.

        :param x: the log of the probability function will be computed in x.
        :type x: ``numpy.ndarray``
        :return: logpdf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self._dist.logpdf(self.theta * x)

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: the cumulative distribution function will be computed in x.
        :type x: ``numpy.ndarray``
        :return: cumulative distribution function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self._dist.cdf(self.loc + self.theta * (x - self.loc))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: point where the natural logarithm of the cumulative distribution function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the cumulative distribution function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self._dist.logcdf(self.theta * x)

    def sf(self, x):
        """
        Survival function, 1 - cumulative distribution function.

        :param x: point where the survival is evaluated.
        :type x: ``int``
        :return: survival function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return 1 - self.cdf(x)

    def logsf(self, x):
        """
        Natural logarithm of the survival function.

        :param x: point where the natural logarithm of the survival function is evaluated.
        :type x: ``int``
        :return: natural logarithm of the survival function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.sf(x))

    def isf(self, x):
        """
        Inverse survival function (inverse of sf).

        :param x: point where the inverse of the survival function is evaluated.
        :type x: ``numpy.ndarray``
        :return: inverse of the survival function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self._dist.isf(self.theta * x)

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size (default is 1).
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)

        return stats.expon.rvs(size=size, random_state=random_state) / self.theta + self.loc

    def entropy(self):
        """
        (Differential) entropy of the RV.

        :return: (differential) entropy.
        :rtype: ``numpy.float64``
        """
        return 1 - np.log(self.theta)

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return 1 / self.theta

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return 1 / self.theta ** 2

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        return np.sqrt(self.var)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of the cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
        """

        hf.assert_type_value(q, 'q', logger, (float, int, np.ndarray, np.floating, np.integer))
        if not isinstance(q, np.ndarray):
            q = np.asarray(q)

        temp = -np.log(1 - q) / self.theta

        zeros = np.where(((q >= 1.) & (q <= 0.)))[0]

        if zeros.size == 0:
            return temp
        else:
            if q.shape == ():
                temp = np.float64('nan')
            else:
                temp[zeros] = np.float64('nan')
            return temp

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        hf.assert_type_value(v, 'v', logger, (np.floating, np.integer, np.ndarray, int, float))
        v = np.array([v]).flatten()
        out = (1 - np.exp(-self.theta * v)) / self.theta
        out[v < 0] = v[v < 0]
        return out
    
    def partial_moment(self, n, low, up):
        """
        Partial moment of order n.

        :param n: moment order.
        :type n: ``int``
        :param low: lower limit of the partial moment.
        :type low: ``int``, ``float``
        :param up: upper limit of the partial moment.
        :type up: ``int``, ``float``
        :return: raw partial moment of order n.
        :rtype: ``float`` 
        """

        hf.assert_type_value(low, 'low', logger, type=(int, float),
        lower_bound=0, upper_bound=float('inf'), upper_close=False)
        hf.assert_type_value(up, 'up', logger, type=(int, float), lower_bound=low, lower_close=False)
        hf.assert_type_value(n, 'n', logger, type=(int, float), lower_bound=0, lower_close=True)
        n = int(n)

        scale = 1/self.theta
        output = scale**n * np.exp(special.loggamma(n+1) - special.loggamma(1))
        output *= stats.gamma(n+1, scale=scale).cdf(up) - stats.gamma(n+1, scale=scale).cdf(low)
        return output


# Gamma
class Gamma(_ContinuousDistribution):
    """
    Gamma distribution.
    When a is an integer it reduces to an Erlang distribution.
    When a=1 it reduces to an Exponential distribution.
    Wrapper to scipy gamma distribution ( ``scipy.stats._continuous_distns.gamma_gen`` ).

    :param scale: scale parameter (inverse of the rate parameter).
    :type scale: ``float``
    :param loc: location parameter.
    :type loc: ``float``
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *a* (``int`` or ``float``) --
          shape parameter a.

    """

    def __init__(self, loc=0, scale=1., **kwargs):
        _ContinuousDistribution.__init__(self)
        self.loc = loc
        self.scale = scale
        self.a = kwargs['a']

    @property
    def a(self):
        return self.__a

    @a.setter
    def a(self, value):
        hf.assert_type_value(value, 'a', logger, (float, int),
                        lower_bound=0, lower_close=False)
        self.__a = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (float, int))
        self.__loc = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        hf.assert_type_value(value, 'scale', logger, (float, int), lower_bound=0, lower_close=False)
        self.__scale = value

    @property
    def rate(self):
        return 1/self.scale

    @property
    def _dist(self):
        return stats.gamma(a=self.a, loc=self.loc, scale=self.scale)

    @staticmethod
    def name():
        return 'gamma'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        hf.assert_type_value(v, 'v', logger, (np.floating, np.integer, np.ndarray, int, float))
        v = np.array([v]).flatten()

        beta = 1 / self.scale
        alpha = self.a

        out = v.copy().astype(np.float64)
        out[v == np.inf] = self.mean()
        flt = (v > 0) * (v < np.inf)
        out[flt] = (alpha / beta) * special.gammainc(alpha + 1, beta * v[flt]) + v[flt] * (1 - special.gammainc(alpha, beta * v[flt]))
        return out

    def partial_moment(self, n, low, up):
        """
        Partial moment of order n.

        :param n: moment order.
        :type n: ``int``
        :param low: lower limit of the partial moment.
        :type low: ``int``, ``float``
        :param up: upper limit of the partial moment.
        :type up: ``int``, ``float``
        :return: raw partial moment of order n.
        :rtype: ``float`` 
        """

        hf.assert_type_value(low, 'low', logger, type=(int, float),
        lower_bound=0, upper_bound=float('inf'), upper_close=False)
        hf.assert_type_value(up, 'up', logger, type=(int, float), lower_bound=low, lower_close=False)
        hf.assert_type_value(n, 'n', logger, type=(int, float), lower_bound=0, lower_close=True)
        n = int(n)

        scale = self.scale
        shape = self.a
        output = scale**n * np.exp(special.loggamma(shape + n) - special.loggamma(shape))
        output *= stats.gamma(shape + n, scale=scale).cdf(up) - stats.gamma(shape + n, scale=scale).cdf(low)
        return output


# Inverse Gamma
class InvGamma(_ContinuousDistribution):
    """
    Wrapper to scipy inverse gamma distribution.
    ``scipy.stats._continuous_distns.invgamma_gen object``

    :param scale: scale parameter.
    :type scale: ``float``
    :param loc: location parameter.
    :type loc: ``float``
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *a* (``int`` or ``float``) --
          shape parameter a.
    """

    def __init__(self, loc=0, scale=1, **kwargs):
        _ContinuousDistribution.__init__(self)
        self.a = kwargs['a']
        self.scale = scale
        self.loc = loc

    @property
    def a(self):
        return self.__a

    @a.setter
    def a(self, value):
        hf.assert_type_value(value, 'a', logger, (float, int), lower_bound=0)
        self.__a = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        hf.assert_type_value(value, 'scale', logger, (float, int), lower_bound=0)
        self.__scale = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (float, int))
        self.__loc = value

    @property
    def _dist(self):
        return stats.invgamma(
            a=self.a,
            loc=self.loc,
            scale=self.scale
        )

    @staticmethod
    def name():
        return 'invgamma'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        hf.assert_type_value(v, 'v', logger, (np.floating, np.integer, np.ndarray, int, float))
        v = np.array([v]).flatten()
        output = v.copy().astype(np.float64)

        output[v == np.inf] = self.mean()
        flt = (v > 0) * (v < np.inf)
        output[flt] = v[flt] * special.gammainc(self.a, self.scale / v[flt]) + self.scale * (
                1 - special.gammainc(self.a - 1, self.scale / v[flt])) * special.gamma(
            self.a - 1) / special.gamma(self.a)
        return output


# Generalized Pareto
class GenPareto(_ContinuousDistribution):
    """
    Wrapper to scipy genpareto distribution.
    When c (i.e. shape) = 0, it reduces to an Exponential distribution.
    When c (i.e. shape) = -1, it reduces to a uniform distribution.
    When the correct parametrization is adopted, it is possible to fit all the Pareto types.
    scipy reference distribution: ``scipy.stats._continuous_distns.genpareto_gen`` .

    :param scale: scale parameter.
    :type scale: ``float``
    :param loc: location parameter.
    :type loc: ``float``
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *c* (``int`` or ``float``) --
          shape parameter.

    """

    def __init__(self, loc=0, scale=1., **kwargs):
        _ContinuousDistribution.__init__(self)
        self.c = kwargs['c']
        self.scale = scale
        self.loc = loc

    @property
    def c(self):
        return self.__c

    @c.setter
    def c(self, value):
        hf.assert_type_value(value, 'c', logger, (float, int))
        self.__c = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        hf.assert_type_value(value, 'scale', logger, (float, int))
        self.__scale = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (float, int))
        self.__loc = value

    @property
    def _dist(self):
        return stats.genpareto(c=self.c, loc=self.loc, scale=self.scale)

    @staticmethod
    def name():
        return 'genpareto'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        hf.assert_type_value(v, 'v', logger, (np.floating, np.integer, np.ndarray, int, float))
        v = np.array([v]).flatten()
        output = v.copy().astype(np.float64)
        output[v == np.inf] = self.mean()
        flt = (v > 0) * (v < np.inf)
        output[flt] = (self.scale / (self.c - 1)) * ((1 + self.c * v[flt] / self.scale) ** (1 - 1 / self.c) - 1)
        return output


# Pareto 2
class Pareto2(GenPareto):
    """
    Pareto TypeII distribution.
    This is a Genpareto distribution with parameter loc = min; scale = scale/shape and c = 1/shape.
    See ``scipy.stats._continuous_distns.genpareto``.

    :param min: location parameter.
    :type min: ``float``
    :param scale: scale parameter.
    :type scale: ``float``
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *shape* (``int`` or ``float``) --
            shape parameter.

    """

    def __init__(self, min=0, scale=1, **kwargs):
        GenPareto.__init__(
            self,
            loc=min,
            scale=scale/kwargs['shape'],
            c=1/kwargs['shape']
            )

    @property
    def shape(self):
        return self.__shape

    @shape.setter
    def shape(self, value):
        hf.assert_type_value(value, 'shape', logger, (float, int))
        self.__shape = value

    @property
    def min(self):
        return self.__min

    @min.setter
    def min(self, value):
        hf.assert_type_value(value, 'min', logger, (float, int))
        self.__min = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        hf.assert_type_value(value, 'scale', logger, (float, int))
        self.__scale = value

    @staticmethod
    def name():
        return 'pareto2'


# Pareto 1
class Pareto1(Pareto2):
    """
    Single-parameter Pareto distribution.
    This is a Pareto II distribution with parameter scale = min.
    ``scipy.stats._continuous_distns.genpareto``

    :param min: pareto 1 location parameter.
    :type min: ``float``
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *shape* (``int`` or ``float``) --
            shape parameter.

    """

    def __init__(self, min=0, **kwargs):
        Pareto2.__init__(
            self,
            loc=min,
            scale=min/kwargs['shape'],
            c=1/kwargs['shape']
            )

    @property
    def shape(self):
        return self.__shape

    @shape.setter
    def shape(self, value):
        hf.assert_type_value(value, 'shape', logger, (float, int))
        self.__shape = value

    @property
    def min(self):
        return self.__min

    @min.setter
    def min(self, value):
        hf.assert_type_value(value, 'min', logger, (float, int))
        self.__min = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        hf.assert_type_value(value, 'scale', logger, (float, int))
        self.__scale = value

    @staticmethod
    def name():
        return 'pareto1'


# Lognormal
class Lognormal(_ContinuousDistribution):
    """
    Lognormal distribution.
    The more common parametrization lognormal(mu, sigma), where mu and sigma are the parameter of the underlying Normal distribution,
    is equivalent to lognormal(log(scale), shape).
    scipy reference distribution: ``scipy.stats._continuous_distns.lognorm_gen``

    :param scale: lognormal scale parameter. The natural logarithm of scale is the mean of the underlying Normal distribution.
    :type scale: ``float``
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *shape* (``int`` or ``float``) --
          shape parameter. It corresponds to the standard deviation of the underlying Normal distribution.
    """

    def __init__(self, scale=1., **kwargs):
        _ContinuousDistribution.__init__(self)
        self.shape = kwargs['shape']
        self.scale = scale

    @property
    def shape(self):
        return self.__shape

    @shape.setter
    def shape(self, value):
        hf.assert_type_value(value, 'shape', logger, (float, int))
        self.__shape = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        hf.assert_type_value(value, 'scale', logger, (float, int))
        self.__scale = value

    @property
    def _dist(self):
        return stats.lognorm(s=self.shape, loc=0, scale=self.scale)

    @property
    def mu(self):
        return np.log(self.scale)
    
    @property
    def sigma(self):
        return self.shape

    @staticmethod
    def name():
        return 'lognormal'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        hf.assert_type_value(v, 'v', logger, (np.floating, np.integer, np.ndarray, int, float))
        v = np.array([v]).flatten()
        out = v.copy().astype(np.float64)
        loc = np.log(self.scale)
        out[v == np.inf] = self.mean()
        flt = (v > 0) * (v < np.inf)
        out[flt] = np.exp(loc + self.shape ** 2 / 2) * (
            stats.norm.cdf((np.log(v[flt]) - (loc + self.shape ** 2)) / self.shape)) + v[flt] * (
                             1 - stats.norm.cdf((np.log(v[flt]) - loc) / self.shape))
        return out

    def partial_moment(self, n, low, up):
        """
        Partial moment of order n.

        :param n: moment order.
        :type n: ``int``
        :param low: lower limit of the partial moment.
        :type low: ``int``, ``float``
        :param up: upper limit of the partial moment.
        :type up: ``int``, ``float``
        :return: raw partial moment of order n.
        :rtype: ``float`` 
        """

        hf.assert_type_value(low, 'low', logger, type=(int, float),
        lower_bound=0, upper_bound=float('inf'), upper_close=False)
        hf.assert_type_value(up, 'up', logger, type=(int, float), lower_bound=low, lower_close=False)
        hf.assert_type_value(n, 'n', logger, type=(int, float), lower_bound=0, lower_close=True)
        n = int(n)

        low_tr = (np.log(low) - self.mu - n * self.sigma**2) / (self.sigma * np.sqrt(2))
        up_tr = (np.log(up) - self.mu - n * self.sigma**2) / (self.sigma * np.sqrt(2))
        out1 = 0.5 * np.exp(n * self.mu + (n * self.sigma)**2/2) 
        out2 = special.erf(up_tr) - special.erf(low_tr)
        return out1 * out2


# Generalized beta
class GenBeta:
    """
    Generalized Beta (GB) distribution, also refer to as Generalized Beta
    of the second kind, or the Generalized Beta Prime distribution.
    If X is a GB distributed r.v., its cumulative distribution function can
    be expressed as:

    Pr[X <= x] = Pr[Y <= (x/scale)^shape3], 0 < x < scale,

    where Y has a Beta distribution, with parameters shape1 and shape2.
    Refer to Appendix A of Klugman, Panjer & Willmot, Loss Models, Wiley.

    """

    def __init__(self, shape1, shape2, shape3, scale=1.):
        self.shape1 = shape1
        self.shape2 = shape2
        self.shape3 = shape3
        self.scale = scale

    @property
    def shape1(self):
        return self.__shape1

    @shape1.setter
    def shape1(self, value):
        hf.assert_type_value(value, 'shape1', logger, (float, int), lower_bound=0, lower_close=False)
        self.__shape1 = value

    @property
    def shape2(self):
        return self.__shape2

    @shape2.setter
    def shape2(self, value):
        hf.assert_type_value(value, 'shape2', logger, (float, int), lower_bound=0, lower_close=False)
        self.__shape2 = value

    @property
    def shape3(self):
        return self.__shape3

    @shape3.setter
    def shape3(self, value):
        hf.assert_type_value(value, 'shape3', logger, (float, int), lower_bound=0, lower_close=False)
        self.__shape3 = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        hf.assert_type_value(value, 'scale', logger, (float, int), lower_bound=0, lower_close=False)
        self.__scale = value

    @property
    def _dist(self):
        return stats.beta(self.shape1, self.shape2)

    @staticmethod
    def category():
        return {'severity'}

    @staticmethod
    def name():
        return 'genbeta'

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size (default is 1).
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """

        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)

        tmp_ = stats.beta(a=self.shape1, b=self.shape2).rvs(size=size, random_state=random_state)
        return self.scale * pow(tmp_, 1.0 / self.shape3)

    def pdf(self, x):
        """
        Probability density function.

        :param x: quantile where probability density function is evaluated.
        :type x: ``numpy.ndarray``, ``list``, ``float``, ``int``

        :return: probability density function in x.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """

        x = hf.arg_type_handler(x)
        x_shape = len(x)
        output = np.zeros(x_shape)

        filter_one = (x == 0.0)
        if np.any(filter_one):
            output[filter_one * (self.shape1 * self.shape3 < 1)] = np.infty
            output[filter_one * (self.shape1 * self.shape3 == 1)] = self.shape3 / special.beta(self.shape1,
                                                                                                     self.shape2)

        filter_two = (x > 0.0) * (x < self.scale)
        if np.any(filter_two):
            x_ = x[filter_two]
            logu = self.shape3 * (np.log(x_) - np.log(self.scale))
            log1mu = np.log1p(-np.exp(logu))
            tmp = np.exp(np.log(self.shape3) + self.shape1 * logu + (self.shape2 - 1.0) * log1mu - np.log(
                x_) - special.betaln(self.shape1, self.shape2))
            output[filter_two] = tmp

        filter_three = (x > 0.0) * (x == self.scale)
        if np.any(filter_three):
            output[filter_three * (self.shape2 < 1)] = np.infty
            output[filter_three * (self.shape2 == 1)] = self.shape1 * self.shape3

        if len(output) == 1:
            output = output.item()

        return output

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: cumulative distribution function will be computed in x.
        :type x: ``float``
        :return: cumulative distribution function in x.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        x = hf.arg_type_handler(x)
        x_shape = len(x)
        output = np.zeros(x_shape)

        filter_one = (x > 0.0) * (x < self.scale)
        if np.any(filter_one):
            u = np.exp(self.shape3 * (np.log(x[filter_one]) - np.log(self.scale)))
            output[filter_one] = self._dist.cdf(u)

        filter_two = (x >= self.scale)
        if np.any(filter_two):
            output[filter_two] = 1

        if len(output) == 1:
            output = output.item()
        return output

    def logpdf(self, x):
        """
        Natural logarithm of the probability distribution function.

        :param x: natural logarithm of the probability distribution function computed in x.
        :type x: ``float``
        :return: natural logarithm of the probability distribution function computed in x.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return np.log(self.pdf(x))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: natural logarithm of the cumulative distribution function computed in x.
        :type x: ``float``
        :return: natural logarithm of the cumulative distribution function in x.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.cdf(x))

    def sf(self, x):
        """
        Survival function, 1 - cumulative distribution function.

        :param x: survival function will be computed in x.
        :type x: ``float``
        :return: survival function in x
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return 1 - self.cdf(x)

    def logsf(self, x):
        """
        Natural logarithm of the survival function.

        :param x: natural logarithm of the survival function computed in x.
        :type x: ``float``
        :return: natural logarithm of the survival function in x
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.sf(x=x))

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
        """
        return self.scale * pow(self._dist.ppf(q=q), 1.0 / self.shape3)

    def isf(self, q):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :return: inverse sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self._dist.ppf(1 - q)

    def moment(self, n):
        """
        Non-central moment of order n.

        :param n: moment order.
        :type n: ``int``
        :return: raw moment of order n.
        :rtype: ``float``
        """

        hf.assert_type_value(n, 'n', logger, (float, int), lower_bound=1, lower_close=True)
        if (n < self.shape1 * self.shape3) and (n < -self.shape2 * self.shape3):
            return np.inf
        tmp_ = n / self.shape3

        return pow(self.scale, n) * special.beta(self.shape1 + tmp_, self.shape2) / special.beta(
            self.shape1, self.shape2)

    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        t_ = []
        if 'm' in moments:
            t_.append(self.mean())
        if 'v' in moments:
            t_.append(self.var())
        if 's' in moments:
            t_.append(self.moment(3) / self.moment(2) ** (3 / 2))
        if 'k' in moments:
            t_.append(self.moment(4) / self.moment(2) ** 2 - 3)
        try:
            assert len(t_) != 0, logger.error("moments argument is not composed of letters 'mvsk'")
        except AssertionError as msg:
            print(msg)

        return tuple(t_)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: ``numpy.float64``
        """
        return self.ppf(0.5)

    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``numpy.float64``
        """
        return self.moment(1)

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``numpy.float64``
        """
        return self.moment(2) - self.moment(1) ** 2

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``numpy.float64``
        """
        return self.var() ** (1 / 2)
  
    def skewness(self):
        """
        Skewness (third standardized moment).

        :return: skewness.
        :rtype: ``float``
        """
        return self.stats(moments='s')
    
    def kurtosis(self):
        """
        Excess kurtosis.

        :return: Excess kurtosis.
        :rtype: ``float``
        """
        return self.stats(moments='k')

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """

        v = hf.arg_type_handler(v)
        v_shape = len(v)
        output = v.copy().astype(np.float64)
        output[v == np.inf] = self.mean()
        filter_ = (v > 0) * (v < np.inf)
        if np.any(filter_):
            v_ = v[filter_]
            z_ = v_.copy().astype(np.float64)
            z_[np.isinf(v_)] = 0
            tmp_ = 1 / self.shape3
            u_ = np.exp(self.shape3 * (np.log(v_) - np.log(self.scale)))

            output[filter_] = self.scale * special.beta(self.shape1 + tmp_, self.shape2) / special.beta(
                self.shape1, self.shape2) * stats.beta.cdf(u_, self.shape1 + tmp_, self.shape2) + z_ * (
                                      1 - self._dist.cdf(u_))

        if 1 <= (- self.shape1 * self.shape3):
            output = [np.infty] * v_shape

        if len(output) == 1:
            output = output.item()

        return output

    def censored_moment(self, n, d, c):
        """
        Non-central moment of order n of the transformed random variable min(max(x - d, 0), c).
        When n = 1 it is the so-called stop loss transformation function.
        
        :param d: deductible, or attachment point.
        :type d: ``int``, ``float``
        :param c: cover, deductible + cover is the detachment point.
        :type c: ``int``, ``float``
        :param n: moment order.
        :type n: ``int``
        :return: raw moment of order n.
        :rtype: ``float``
        """
        return hf.censored_moment(n=n, d=d, c=c, dist=self)
    
    def partial_moment(self, n, low, up):
        """
        Non-central partial moment of order n.

        :param n: moment order.
        :type n: ``int``
        :param low: lower limit of the partial moment.
        :type low: ``int``, ``float``
        :param up: upper limit of the partial moment.
        :type up: ``int``, ``float``
        :return: raw partial moment of order n.
        :rtype: ``float`` 
        """
        return hf.partial_moment(n=n, low=low, up=up, dist=self) 
    
    def truncated_moment(self, n, low, up):
        """
        Non-central truncated moment of order n.

        :param n: moment order.
        :type n: ``int``
        :param low: lower truncation point.
        :type low: ``int``, ``float``
        :param up: upper truncation point.
        :type up: ``int``, ``float``
        :return: raw truncated moment of order n.
        :rtype: ``float`` 
        """
        adj = self.cdf(up) - self.cdf(low)
        return hf.partial_moment(n=n, low=low, up=up, dist=self) / adj


# Burr
class Burr12(_ContinuousDistribution):
    """
    Burr distribution, also referred to as the Burr Type XII, Singh–Maddala distribution.
    When d=1, this is a Fisk distribution.
    When c=d, this is a Paralogistic distribution.
    scipy reference distribution: ``scipy.stats._continuous_distns.burr_gen object``

    :param scale: burr scale parameter.
    :type scale: ``float``
    :param loc: burr location parameter.
    :type loc: ``float``
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *c* (``int`` or ``float``) --
          shape parameter c.
        * *d* (``int`` or ``float``) --
          shape parameter d.
    """

    def __init__(self, loc=0, scale=1, **kwargs):
        _ContinuousDistribution.__init__(self)
        self.c = kwargs['c']
        self.d = kwargs['d']
        self.scale = scale
        self.loc = loc

    @property
    def c(self):
        return self.__c

    @c.setter
    def c(self, value):
        hf.assert_type_value(value, 'c', logger, (float, int))
        self.__c = value

    @property
    def d(self):
        return self.__d

    @d.setter
    def d(self, value):
        hf.assert_type_value(value, 'd', logger, (float, int))
        self.__d = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (float, int))
        self.__loc = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        hf.assert_type_value(value, 'scale', logger, (float, int))
        self.__scale = value

    @property
    def _dist(self):
        return stats.burr12(c=self.c, d=self.d, loc=self.loc, scale=self.scale)

    @staticmethod
    def name():
        return 'burr12'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        hf.assert_type_value(v, 'v', logger, (np.floating, np.integer, np.ndarray, int, float))
        v = np.array([v]).flatten()
        output = v.copy().astype(np.float64)
        u = v.copy().astype(np.float64)
        output[v == np.inf] = self.mean()
        flt = (v > 0) * (v < np.inf)
        u[flt] = 1 / (1 + (v[flt] / self.scale) ** self.c)
        temp = self.scale * special.gamma(1 + 1 / self.c) * special.gamma(
            self.d - 1 / self.c) / special.gamma(self.d)
        output[flt] = v[flt] * (u[flt] ** self.d) + special.betaincinv(
            1 + 1 / self.c, self.d - 1 / self.c, 1 - u[flt]) * temp
        return output


# Paralogistic
class Paralogistic(Burr12):
    """
            Paralogistic distribution.
            This is a Burr12 distribution with same parameters.
            ``scipy.stats._continuous_distns.burr12``

            :param scale: paralogistic scale parameter.
            :type scale: ``float``
            :param loc: paralogistic location parameter.
            :type loc: ``float``
            :param \\**kwargs:
                See below

            :Keyword Arguments:
                * *a* (``int`` or ``float``) --
                  distribution parameter a.

            """

    def __init__(self, loc=0, scale=1, **kwargs):
        Burr12.__init__(self,
                        loc=loc,
                        scale=scale,
                        c=kwargs['a'],
                        d=kwargs['a'])

    @property
    def a(self):
        return self.__a

    @a.setter
    def a(self, value):
        hf.assert_type_value(value, 'a', logger, (float, int))
        self.__a = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (float, int))
        self.__loc = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        hf.assert_type_value(value, 'scale', logger, (float, int))
        self.__scale = value

    @staticmethod
    def name():
        return 'paralogistic'


# Dagum
class Dagum(_ContinuousDistribution):
    """
    Wrapper to scipy mielke distribution.
    It is referred to the Inverse Burr, Mielke Beta-Kappa.
    When d=s, this is an inverse paralogistic.
    ``scipy.stats._continuous_distns.mielke``

    :param scale: dagum scale parameter.
    :type scale: ``float``
    :param loc: dagum location parameter.
    :type loc: ``float``
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *d* (``int`` or ``float``) --
          shape parameter d.
        * *s* (``int`` or ``float``) --
          shape parameter s.
    """

    def __init__(self, loc=0, scale=1, **kwargs):
        _ContinuousDistribution.__init__(self)
        self.d = kwargs['d']
        self.s = kwargs['s']
        self.scale = scale
        self.loc = loc

    @property
    def d(self):
        return self.__d

    @d.setter
    def d(self, value):
        hf.assert_type_value(value, 'd', logger, (float, int), lower_bound=0)
        self.__d = value

    @property
    def s(self):
        return self.__s

    @s.setter
    def s(self, value):
        hf.assert_type_value(value, 's', logger, (float, int), lower_bound=0)
        self.__s = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        hf.assert_type_value(value, 'scale', logger, (float, int), lower_bound=0)
        self.__scale = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (float, int))
        self.__loc = value

    @property
    def k(self):
        return self.d * self.s

    @property
    def _dist(self):
        return stats.mielke(
            k=self.k,
            s=self.s,
            loc=self.loc,
            scale=self.scale
        )

    @staticmethod
    def name():
        return 'dagum'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        hf.assert_type_value(v, 'v', logger, (np.floating, np.integer, np.ndarray, int, float))
        v = np.array([v]).flatten()
        output = v.copy().astype(np.float64)
        u = v.copy().astype(np.float64)
        output[v == np.inf] = self.mean()
        flt = (v > 0) * (v < np.inf)
        u[flt] = (v[flt] / self.scale) ** self.s / (1 + (v[flt] / self.scale) ** self.s)
        output[flt] = v[flt] * (1 - u[flt] ** self.d) + special.betaincinv(
            self.d + 1 / self.s, 1 - 1 / self.s, u[flt]) * (
            self.scale * special.gamma(self.d + 1 / self.s) * special.gamma(1 - 1 /self.s) /
            special.gamma(self.d))
        return output


# Inverse Paralogistic
class InvParalogistic(Dagum):

    """
    Inverse paralogistic distribution.
    This is a Dagum distribution with same parameters.
    ``scipy.stats._continuous_distns.mielke``

    :param scale: inverse paralogistic scale parameter.
    :type scale: ``float``
    :param loc: inverse paralogistic location parameter.
    :type loc: ``float``
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *b* (``int`` or ``float``) --
            distribution parameter b.
    """

    def __init__(self, loc=0, scale=1, **kwargs):

        Dagum.__init__(self,
                        loc=loc,
                        scale=scale,
                        d =kwargs['b'],
                        s =kwargs['b'])

    @property
    def b(self):
        return self.__b

    @b.setter
    def b(self, value):
        hf.assert_type_value(value, 'b', logger, (float, int), lower_bound=0)
        self.__b = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (float, int))
        self.__loc = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        hf.assert_type_value(value, 'scale', logger, (float, int))
        self.__scale = value

    @staticmethod
    def name():
        return 'invparalogistic'


# Weibull
class Weibull(_ContinuousDistribution):
    """
    Wrapper to scipy Weibull (Weibull_min) distribution.
    ``scipy.stats._continuous_distns.weibull_min_gen object``

    :param scale: scale parameter.
    :type scale: ``float``
    :param loc: location parameter.
    :type loc: ``float``
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *c* (``int`` or ``float``) --
          shape parameter c.

    """

    def __init__(self, loc=0, scale=1, **kwargs):
        _ContinuousDistribution.__init__(self)
        self.c = kwargs['c']
        self.scale = scale
        self.loc = loc

    @property
    def c(self):
        return self.__c

    @c.setter
    def c(self, value):
        hf.assert_type_value(value, 'c', logger, (float, int), lower_bound=0)
        self.__c = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        hf.assert_type_value(value, 'scale', logger, (float, int), lower_bound=0)
        self.__scale = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (float, int))
        self.__loc = value

    @property
    def _dist(self):
        return stats.weibull_min(
            c=self.c,
            loc=self.loc,
            scale=self.scale
        )

    @staticmethod
    def name():
        return 'weibull'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        hf.assert_type_value(v, 'v', logger, (np.floating, np.integer, np.ndarray, int, float))
        v = np.array([v]).flatten()
        output = v.copy().astype(np.float64)
        output[v == np.inf] = self.mean()
        flt = (v > 0) * (v < np.inf)
        output[flt] = v[flt] * np.exp(-(v[flt] / self.scale) ** self.c) + self.scale * special.gamma(
            1 + 1 / self.c) * special.gammainc(1 + 1 / self.c, (v[flt] / self.scale) ** self.c)
        return output


# Inverse Weibull
class InvWeibull(_ContinuousDistribution):
    """
    Wrapper to scipy inverse Weibull distribution.
    ``scipy.stats._continuous_distns.invweibull_gen object``

    :param scale: scale parameter.
    :type scale: ``float``
    :param loc: location parameter.
    :type loc: ``float``
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *c* (``int`` or ``float``) --
          shape parameter c.
    """

    def __init__(self, loc=0, scale=1, **kwargs):
        _ContinuousDistribution.__init__(self)
        self.c = kwargs['c']
        self.scale = scale
        self.loc = loc

    @property
    def c(self):
        return self.__c

    @c.setter
    def c(self, value):
        hf.assert_type_value(value, 'c', logger, (float, int), lower_bound=0)
        self.__c = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        hf.assert_type_value(value, 'scale', logger, (float, int), lower_bound=0)
        self.__scale = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (float, int))
        self.__loc = value

    @property
    def _dist(self):
        return stats.invweibull(
            c=self.c,
            loc=self.loc,
            scale=self.scale
        )

    @staticmethod
    def name():
        return 'invweibull'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        hf.assert_type_value(v, 'v', logger, (np.floating, np.integer, np.ndarray, int, float))
        v = np.array([v]).flatten()
        output = v.copy().astype(np.float64)
        output[v == np.inf] = self.mean()
        flt = (v > 0) * (v < np.inf)
        output[flt] = v[flt] * (1 - np.exp(-(self.scale / v[flt]) ** self.c)) + self.scale * special.gamma(
            1 - 1 / self.c) * (1 - special.gammainc(1 - 1 / self.c, (self.scale / v[flt]) ** self.c))
        return output


# Inverse Gaussian
class InvGauss(_ContinuousDistribution):
    """
    Wrapper to scipy inverse gaussian distribution.
    ``scipy.stats._continuous_distns.invgauss_gen object``

    :param scale: scale parameter.
    :type scale: ``float``
    :param loc: location parameter.
    :type loc: ``float``
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *mu* (``int`` or ``float``) --
          shape parameter mu.
    """

    def __init__(self, loc=0., scale=1., **kwargs):
        _ContinuousDistribution.__init__(self)
        self.scale = scale
        self.loc = loc
        self.mu = kwargs['mu']

    @property
    def mu(self):
        return self.__mu

    @mu.setter
    def mu(self, value):
        hf.assert_type_value(value, 'mu', logger, (float, int))
        self.__mu = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (float, int))
        self.__loc = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        hf.assert_type_value(value, 'scale', logger, (float, int))
        self.__scale = value

    @property
    def _dist(self):
        return stats.invgauss(mu=self.mu, loc=self.loc, scale=self.scale)

    @staticmethod
    def name():
        return 'invgauss'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        hf.assert_type_value(v, 'v', logger, (np.floating, np.integer, np.ndarray, int, float))
        v = np.array([v]).flatten()
        output = v.copy().astype(np.float64)
        z = v.copy().astype(np.float64)
        y = v.copy().astype(np.float64)

        output[v == np.inf] = self.mean()
        flt = (v > 0) * (v < np.inf)

        z[flt] = (v[flt] - self.mu) / self.mu
        y[flt] = (v[flt] + self.mu) / self.mu
        output[flt] = v[flt] - self.mu * z[flt] * stats.norm.cdf(z[flt] * np.sqrt(
            1 / v[flt])) - self.mu * y[flt] * np.exp(2 / self.mu) * stats.norm.cdf(
            -y[flt] * np.sqrt(1 / v[flt]))
        return self.scale * output


# Fisk
class Fisk(_ContinuousDistribution):
    """
    Wrapper to scipy Fisk distribution.
    ``scipy.stats._continuous_distns.fisk_gen object``

    :param scale: scale parameter.
    :type scale: ``float``
    :param loc: location parameter.
    :type loc: ``float``
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *c* (``int`` or ``float``) --
          shape parameter c.
    """

    def __init__(self, loc=0., scale=1., **kwargs):
        _ContinuousDistribution.__init__(self)
        self.scale = scale
        self.loc = loc
        self.c = kwargs['c']

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (float, int))
        self.__loc = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        hf.assert_type_value(value, 'scale', logger, (float, int))
        self.__scale = value

    @property
    def c(self):
        return self.__c

    @c.setter
    def c(self, value):
        hf.assert_type_value(value, 'c', logger, (float, int), lower_bound=0)
        self.__c = value

    @property
    def _dist(self):
        return stats.fisk(c=self.c, loc=self.loc, scale=self.scale)

    @staticmethod
    def name():
        return 'fisk'

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        hf.assert_type_value(v, 'v', logger, (np.floating, np.integer, np.ndarray, int, float))
        v = np.array([v]).flatten()
        output = v.copy().astype(np.float64)
        output[v == np.inf] = self.mean()
        flt = (v > 0) * (v < np.inf)
        u = v.copy().astype(np.float64)
        u[flt] = (v[flt] ** self.c) / (1 + v[flt] ** self.c)
        output[flt] = v[flt] * (1 - u[flt]) + self.scale * special.gamma(
            1 + 1 / self.c) * special.gamma(1 - 1 / self.c) * special.betaincinv(
            1 + 1 / self.c, 1 - 1 / self.c, u[flt])
        return output


# Piecewise-Linear
class PWL:
    """
    Piecewise-linear distribution (a.k.a. mixture of continguous uniform distribution).
    Distribution specified by a set of points which are delimiting the intervals and
    a set of cumulative probabilities (associated to the intervals).
    Between two consecutive points, the random variable is uniformly distributed.
    Points must be non negatives.
    
    :param points: limit points of the bins of the distribution. At least the two points where cumulative probabilities are 0 and 1 needs to be provided. Its length must match ``cumprobs`` one.
    :type points: ``list`` or ``np.ndarray`` of ``int`` or ``float``
    :param cumprobs: cumulative probabilities associated to the points. At least cumulative probabilities of 0 and 1 needs to be provided. Its length must match ``points`` one.
    :type cumprobs: ``list`` or ``np.ndarray`` of ``int`` or ``float``
    """
    def __init__(self, points, cumprobs):
        self.points = points
        self.cumprobs = cumprobs
        self._check_points_cumprobs_length()
    
    @staticmethod
    def name():
        return 'pwl'

    @staticmethod
    def category():
        return {'severity'}

    @property
    def cumprobs(self):
        return self.__cumprobs
    
    @cumprobs.setter
    def cumprobs(self, value):
        hf.assert_type_value(value, 'cumprobs', logger, (list, np.ndarray))
        hf.check_condition(len(value), 2, 'cumprobs length', logger, '>=')
        value = np.array(value).flatten()
        if np.any(value > 1):
            message = 'Make sure cumprobs is lower than or equal to 1'
            logger.error(message)
            raise ValueError(message)
        if np.any(value < 0):
            message = 'Make sure cumprobs is higher than or equal to 0'
            logger.error(message)
            raise ValueError(message)
        hf.check_condition(
            np.all(value[1:] >= value[:-1]),
            True,
            'assertion cumprobs increasing',
            logger
        )
        hf.check_condition(
            value[0], 0, 'first cumulative probability', logger
        )
        hf.check_condition(
            value[-1], 1, 'last cumulative probability', logger
        )
        self.__cumprobs = value
    
    @property
    def points(self):
        return self.__points
    
    @points.setter
    def points(self, value):
        hf.assert_type_value(value, 'points', logger, (list, np.ndarray))
        value = np.array(value).flatten()
        hf.check_condition(len(value), 2, 'points length', logger, '>=')
        hf.assert_type_value(
            value[0], 'points', logger, (np.floating, np.integer, float, int)
        )
        hf.check_condition(
            np.all(value[1:] >= value[:-1]),
            True,
            'assertion points increasing',
            logger
        )
        self.__points = value
    
    @property
    def max(self):
        return np.max(self.points)

    @property
    def min(self):
        return np.min(self.points)
    
    @property
    def upoints(self):
        return self.points[1:]

    @property
    def lpoints(self):
        return self.points[:-1]

    @property
    def ranges(self):
        output = self.upoints - self.lpoints
        output[output == 0] = 1
        return output

    @property
    def weights(self):
        return np.diff(self.cumprobs)

    def _check_points_cumprobs_length(self):
        hf.check_condition(
            len(self.cumprobs),
            len(self.points),
            'cumprobs length',
            logger
        )
    
    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int`` or ``float``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        isscalar = not isinstance(x, (np.ndarray, list)) 
        x = np.array(x).reshape(-1, 1)

        dens = self.weights / self.ranges
        x_= np.minimum(
                np.maximum(
                    np.subtract(x, self.lpoints.reshape(1, -1)),
                    0),
            self.ranges.reshape(1, -1))
        output = np.sum(x_ * dens.reshape(1, -1), axis=1)
        
        if isscalar:
            return output.item()
        else:
            return output
    
    def pdf(self, x):
        """
        Probability density function.

        :param x: quantile where probability density function is evaluated.
        :type x: ``numpy.ndarray``, ``list``, ``float``, ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        isscalar = not isinstance(x, (np.ndarray, list)) 
        x = np.array(x).reshape(-1, 1)
        output = np.empty(len(x))

        # cases where x is within the interval
        indx = np.logical_and(x < self.upoints, x > self.lpoints)
        filter = np.any(indx, axis=1)
        output[filter] = np.sum(self.weights.reshape(1, -1) * indx / self.ranges.reshape(1, -1) * indx, axis=1)[filter]
        # cases where x is equal to lower bound or to max
        indx = np.logical_or(x == self.lpoints, x == self.max)
        filter = np.any(indx, axis=1)
        output[filter] = np.sum(self.weights.reshape(1, -1) * indx / self.ranges.reshape(1, -1) * indx, axis=1)[filter]
        # cases where x is equal to a node
        indx = np.logical_and(x == self.upoints, x == self.lpoints)
        filter = np.any(indx, axis=1)
        output[filter] = np.sum(self.weights.reshape(1, -1) * indx / self.ranges.reshape(1, -1) * indx, axis=1)[filter]
        # cases outside the range [min, max]
        indx = np.logical_or(x < self.min, x > self.max)
        filter = np.any(indx, axis=1)
        output[filter] = 0

        if isscalar:
            return output.item()
        else:
            return output

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of the cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
        """
        hf.assert_type_value(
            q, 'q', logger, (np.floating, np.integer, int, float, list, np.ndarray)
            )
        isscalar = not isinstance(q, (np.ndarray, list)) 
        q = np.ravel(q)
        if np.any(q > 1):
            message = 'Make sure q is lower than or equal to 1'
            logger.error(message)
            raise ValueError(message)
        if np.any(q < 0):
            message = 'Make sure q is higher than or equal to 0'
            logger.error(message)
            raise ValueError(message)

        q = np.array(q).reshape(-1, 1)
        
        lower_probs = self.cumprobs[:-1].reshape(1, -1)
        ps_ = np.minimum(
                np.maximum(
                    np.subtract(q, lower_probs) / self.weights.reshape(1, -1),
                    0), 1)
        output = np.sum(ps_ * self.ranges, axis=1) + self.lpoints[0]

        if isscalar:
            return output.item()
        else:
            return output
    
    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator (default=None).
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        output = self.ppf(
            np.random.uniform(low=0, high=1.0, size=int(size))
        )
        return output

    def sf(self, x):
        """
        Survival function, 1 - cumulative distribution function.

        :param x: quantile where the survival function is evaluated.
        :type x: ``int`` or ``float``
        :return: survival function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return 1 - self.cdf(x)

    def moment(self, n=1, central=False):
        """
        Moment of order n.

        :param central: ``True`` if the moment is central, ``False`` if the moment is raw.
        :type central: ``bool``
        :param n: moment order.
        :type n: ``int``
        :return: moment of order n.
        :rtype: ``float``
        """
        hf.assert_type_value(central, 'central', logger, (bool))
        hf.assert_type_value(n, 'n', logger, (int, float), lower_bound=0, lower_close=False)
        n = int(n)
        if central is False:
            means = (self.upoints**(n+1) - self.lpoints**(n+1)) / (self.ranges * (n+1))
            indx = (self.upoints - self.lpoints) == 0
            means[indx] = self.upoints[indx]
            output = np.average(means, weights=self.weights)
        else:
            original_points = self.points.copy().astype(np.float64)
            # shift points with respect to mean
            self.points = self.points - self.moment(n=1)
            output = self.moment(n=n, central=False)
            # restore original points
            self.points = original_points
        return output
    
    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``float``
        """
        return self.moment()
    
    def var(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``float``
        """
        return self.moment(central=True, n=2)

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``float``
        """
        return self.moment(central=True, n=2)**0.5

    def skewness(self):
        """
        Skewness.

        :return: skewness.
        :rtype: ``numpy.float64``
        """
        return self.moment(central=True, n=3) / self.std()**3

    def kurtosis(self, excess=False):
        """
        Kurtosis.
        If excess is ``True``, the excess of kurtosis (Fisher's definition) is calculated,
        i.e. 3.0 is subtracted from the raw result to give 0.0 for a normal distribution.

        :param excess: ``True`` if excess of kurtosis, ``False`` otherwise.
        :type excess: ``bool``
        :return: kurtosis.
        :rtype: ``numpy.float64``
        """
        exc = 3 if excess else 0
        return self.moment(central=True, n=4) / self.var()**2 - exc

    def censored_moment(self, n, d, c):
        """
        Non-central moment of order n of the transformed random variable min(max(x - u, 0), v).
        When n = 1 it is the so-called stop loss transformation function.
        
        :param d: deductible, or attachment point.
        :type d: ``int``, ``float``
        :param c: cover, deductible + cover is the detachment point.
        :type c: ``int``, ``float``
        :param n: moment order.
        :type n: ``int``
        :return: raw moment of order n.
        :rtype: ``float``
        """
        hf.assert_type_value(n, 'n', logger, (int, float), lower_bound=1)
        hf.assert_type_value(c, 'c', logger, (np.floating, np.integer, np.ndarray, int, float))
        hf.assert_type_value(d, 'd', logger, (np.floating, np.integer, np.ndarray, int, float))
        c = np.array([c]).flatten()
        d = np.array([d]).flatten()
        hf.check_condition(
            len(c), len(d), 'v length', logger, '=='
        )
        original_points = self.points.copy().astype(np.float64)
        output = np.empty(len(c))
        for i in range(len(output)):
            self.points = np.minimum(
                np.maximum(original_points - d[i], 0),
                c[i])
            output[i] = self.moment(n=n, central=False)
        
        # restore original points
        self.points = original_points
        if len(output) == 1:
            return output.item()
        else:
            return output

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        hf.assert_type_value(v, 'v', logger, (np.floating, np.integer, np.ndarray, int, float))
        v = np.array([v]).flatten()
        u = np.zeros(len(v))
        output = v.copy().astype(np.float64)
        output[v == np.inf] = self.mean()
        flt = (v > 0) * (v < np.inf)
        output[flt] = self.censored_moment(n=1, d=u[flt], c=v[flt])
        return output


# Piecewise-Costant
class PWC:
    """
    Piecewise-constant distribution (a.k.a. empirical cumulative distribution).
    Distribution specified by a set of nodes and a set of cumulative probabilities (associated to the nodes).
    Nodes must be non negatives.
        
    :param nodes: nodes of the distribution. Its length must match ``cumprobs`` one.
    :type nodes: ``list`` or ``np.ndarray`` of ``int`` or ``float``
    :param cumprobs: cumulative probabilities associated to the nodes. Its length must match ``nodes`` one.
    :type cumprobs: ``list`` or ``np.ndarray`` of ``int`` or ``float``
    :param legit: boolean. If True (default) the distribution must be a legitimate probability, i.e. last cumprobs value equal to 1.
    :type legit: ``bool``
    """
    def __init__(self, nodes, cumprobs, legit=True):
        self.legit = legit
        self.nodes = nodes
        self.cumprobs = cumprobs
        self._check_nodes_cumprobs_length()
    
    @staticmethod
    def name():
        return 'pwc'

    @staticmethod
    def category():
        return {'severity'}

    @property
    def legit(self):
        return self.__legit

    @legit.setter
    def legit(self, value):
        hf.assert_type_value(value, 'legit', logger, (bool))
        self.__legit = value

    @property
    def cumprobs(self):
        return self.__cumprobs

    @cumprobs.setter
    def cumprobs(self, value):
        hf.assert_type_value(value, 'cumprobs', logger, (list, np.ndarray))
        hf.check_condition(len(value), 2, 'cumprobs length', logger, '>=')
        value = np.array(value).flatten()
        if np.any(value > 1):
            message = 'Make sure cumprobs is lower than or equal to 1'
            logger.error(message)
            raise ValueError(message)
        if np.any(value < 0):
            message = 'Make sure cumprobs is higher than or equal to 0'
            logger.error(message)
            raise ValueError(message)
        hf.check_condition(
            np.all(value[1:] >= value[:-1]),
            True,
            'assertion cumprobs non decreasing',
            logger
        )
        if self.legit:
            hf.check_condition(
                value[-1], 1, 'last cumulative probability', logger
            )
        self.__cumprobs = value
    
    @property
    def nodes(self):
        return self.__nodes
    
    @nodes.setter
    def nodes(self, value):
        hf.assert_type_value(value, 'nodes', logger, (list, np.ndarray))
        value = np.array(value).flatten()
        hf.check_condition(len(value), 2, 'nodes length', logger, '>=')
        hf.assert_type_value(
            value[0], 'nodes', logger, (np.floating, np.integer, float, int)
        )        
        hf.check_condition(
            np.all(value[1:] > value[:-1]),
            True,
            'assertion nodes non decreasing',
            logger
        )
        self.__nodes = value
    
    @property
    def max(self):
        return np.max(self.nodes)

    @property
    def min(self):
        return np.min(self.nodes)
    
    @property
    def pmf(self):
        return np.diff(self.cumprobs, prepend=0)

    def _check_nodes_cumprobs_length(self):
        hf.check_condition(
            len(self.cumprobs),
            len(self.nodes),
            'cumprobs length',
            logger
        )
    
    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int`` or ``float``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """  
        isscalar = not isinstance(x, (np.ndarray, list)) 
        x = np.array(x).flatten()
        output = np.empty(x.size)
        output[x < self.min] = 0
        output[x > self.max] = 1
        filter_ = np.logical_and(x >= self.min, x <= self.max)
        x_ = x[filter_]
        index = np.searchsorted(self.nodes, x_, side='right') - 1
        output[filter_] = self.cumprobs[index]
        
        if isscalar:
            return output.item()
        else:
            return output
    
    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of the cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
        """
        hf.assert_type_value(
            q, 'q', logger, (np.floating, np.integer, int, float, list, np.ndarray)
            )
        isscalar = not isinstance(q, (np.ndarray, list)) 
        q = np.ravel(q).reshape(1, -1)
        if np.any(q > 1):
            message = 'Make sure q is lower than or equal to 1'
            logger.error(message)
            raise ValueError(message)
        if np.any(q < 0):
            message = 'Make sure q is higher than or equal to 0'
            logger.error(message)
            raise ValueError(message)
        
        cumprobs = self.cumprobs.reshape(-1, 1).copy().astype(np.float64)
        cumprobs[-1, 0] = 1
        index = np.sum(q > cumprobs, axis=0)
        output = self.nodes[index]
        
        if isscalar:
            return output.item()
        else:
            return output
    
    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator (default=None).
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        output = np.random.choice(
            self.nodes, size=size, p=self.pmf
        )
        return output

    def sf(self, x):
        """
        Survival function, 1 - cumulative distribution function.

        :param x: quantile where the survival function is evaluated.
        :type x: ``int`` or ``float``
        :return: survival function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return 1 - self.cdf(x)

    def moment(self, central=False, n=1):
        """
        Moment of order n.

        :param central: ``True`` if the moment is central, ``False`` if the moment is raw.
        :type central: ``bool``
        :param n: moment order.
        :type n: ``int``
        :return: moment of order n.
        :rtype: ``float``
        """
        hf.assert_type_value(central, 'central', logger, (bool))
        hf.assert_type_value(n, 'n', logger, (int, float), lower_bound=0, lower_close=False)
        n = int(n)
        mean = 0 if not central else self.mean()
        output = np.sum(
            self.pmf * (self.nodes - mean)**n 
            )
        return output
    
    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``float``
        """
        return np.sum(self.nodes * self.pmf)
    
    def var(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``float``
        """
        return self.moment(central=True, n=2)

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``float``
        """
        return self.moment(central=True, n=2)**0.5

    def skewness(self):
        """
        Skewness.

        :return: skewness.
        :rtype: ``numpy.float64``
        """
        return self.moment(central=True, n=3) / self.std()**3

    def kurtosis(self, excess=False):
        """
        Kurtosis.
        If excess is ``True``, the excess of kurtosis (Fisher's definition) is calculated,
        i.e. 3.0 is subtracted from the raw result to give 0.0 for a normal distribution.

        :param excess: ``True`` if excess of kurtosis, ``False`` otherwise.
        :type excess: ``bool``
        :return: kurtosis.
        :rtype: ``numpy.float64``
        """
        exc = 3 if excess else 0
        return self.moment(central=True, n=4) / self.var()**2 - exc

    def censored_moment(self, n, d, c):
        """
        Non-central moment of order n of the transformed random variable min(max(x - u, 0), v).
        When n = 1 it is the so-called stop loss transformation function.
        
        :param d: deductible, or attachment point.
        :type d: ``int``, ``float``
        :param c: cover, deductible + cover is the detachment point.
        :type c: ``int``, ``float``
        :param n: moment order.
        :type n: ``int``
        :return: raw moment of order n.
        :rtype: ``float``
        """
        hf.assert_type_value(n, 'n', logger, (int, float), lower_bound=1)
        hf.assert_type_value(c, 'c', logger, (np.floating, np.integer, np.ndarray, int, float))
        hf.assert_type_value(d, 'd', logger, (np.floating, np.integer, np.ndarray, int, float))
        c = np.array([c]).flatten()
        d = np.array([d]).flatten()
        hf.check_condition(
            len(c), len(d), 'v length', logger, '=='
        )
        
        output = np.minimum(
            np.maximum(self.nodes.reshape(-1, 1) - d.reshape(1, -1), 0),
            c.reshape(1, -1),
            dtype=np.float64)
        output = np.sum(self.pmf.reshape(-1, 1) * output**n, axis=0)
        if len(output) == 1:
            return output.item()
        else:
            return output
        
    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """
        hf.assert_type_value(v, 'v', logger, (np.floating, np.ndarray, int, float, np.integer))
        v = np.array([v]).flatten()
        output = v.copy().astype(np.float64)
        output[v == np.inf] = self.mean()
        flt = (v > 0) * (v < np.inf)
        output[flt] = self.censored_moment(n=1, d=np.repeat(0, len(v[flt])), c=v[flt])
        return output


# Log Gamma
class LogGamma:
    """
    Log Gamma distribution.
    Random variable whose logarithm transformation is Gamma distributed. 
    Parameters (``a`` and ``scale``) refer to the underlying log-transformed random variable.
    Remark: distribution form and parametrization differ from
    <scipy.stats._continuous_distns.loggamma_gen object>

    :param scale: scale parameter (inverse of the rate parameter).
    :type scale: ``float``
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *a* (``int`` or ``float``) --
          shape parameter a.

    """
    def __init__(self, scale=1., **kwargs):
        self.scale = scale
        self.a = kwargs['a']
    
    @staticmethod
    def name():
        return 'loggamma'

    @staticmethod
    def category():
        return {'severity'}

    @property
    def a(self):
        return self.__a

    @a.setter
    def a(self, value):
        hf.assert_type_value(value, 'a', logger, (float, int),
                             lower_bound=0, lower_close=False)
        self.__a = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        hf.assert_type_value(value, 'scale', logger, (float, int),
                             lower_bound=0, lower_close=False)
        self.__scale = value

    @property
    def rate(self):
        return 1 / self.scale

    @property
    def _dist(self):
        return stats.gamma(a=self.a, loc=0, scale=self.scale)

    def pdf(self, x):
        """
        Probability density function.

        :param x: quantile where probability density function is evaluated.
        :type x: ``numpy.ndarray``, ``list``, ``float``, ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.exp(self._dist.logpdf(np.log(x)) - np.log(x))

    def cdf(self, x):
        """
        Cumulative distribution function.

        :param x: quantile where the cumulative distribution function is evaluated.
        :type x: ``int`` or ``float``
        :return: cumulative distribution function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return self._dist.cdf(np.log(x))
    
    def sf(self, x):
        """
        Survival function, 1 - cumulative distribution function.

        :param x: quantile where the survival function is evaluated.
        :type x: ``int`` or ``float``
        :return: survival function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return 1 - self.cdf(x)

    def ppf(self, q):
        """
        Percent point function, a.k.a. the quantile function, inverse of the cumulative distribution function.

        :param q: level at which the percent point function is evaluated.
        :type q: ``float``
        :return: percent point function.
        :rtype: ``numpy.float64`` or ``numpy.int`` or ``numpy.ndarray``
        """
        return np.exp(self._dist.ppf(q))

    def rvs(self, size=1, random_state=None):
        """
        Random variates.

        :param size: random variates sample size (default is 1).
        :type size: ``int``
        :param random_state: random state for the random number generator.
        :type random_state: ``int``

        :return: Random variates.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)

        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)
        return np.exp(self._dist.rvs(size=size, random_state=random_state))
    
    def moment(self, n):
        """
        Non-central moment of order n.

        :param n: moment order.
        :type n: ``int``
        :return: raw moment of order n.
        :rtype: ``float``
        """

        hf.assert_type_value(
            n, 'n', logger, (float, int),
            lower_bound=1,
            lower_close=True)
        n = int(n)
        if n >= self.rate:
            return float('inf')
        else:
            return (1.0 - n / self.rate)**(-self.a)
    
    def mean(self):
        """
        Mean of the distribution.

        :return: mean.
        :rtype: ``float``
        """
        return self.moment(1)

    def var(self):
        """
        Variance of the distribution.

        :return: variance.
        :rtype: ``float``

        """
        return self.moment(2) - self.moment(1)**2

    def std(self):
        """
        Standard deviation of the distribution.

        :return: standard deviation.
        :rtype: ``float``

        """
        return self.var()**(1/2)

    def logpdf(self, x):
        """
        Natural logarithm of the probability distribution function.

        :param x: natural logarithm of the probability distribution function computed in x.
        :type x: ``float``
        :return: natural logarithm of the probability distribution function computed in x.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        return np.log(self.pdf(x))

    def logcdf(self, x):
        """
        Natural logarithm of the cumulative distribution function.

        :param x: natural logarithm of the cumulative distribution function computed in x.
        :type x: ``float``
        :return: natural logarithm of the cumulative distribution function in x.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.cdf(x))

    def logsf(self, x):
        """
        Natural logarithm of the survival function.

        :param x: natural logarithm of the survival function computed in x.
        :type x: ``float``
        :return: natural logarithm of the survival function in x
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return np.log(self.sf(x=x))

    def isf(self, q):
        """
        Inverse survival function (inverse of sf).

        :param q: Inverse survival function computed in q.
        :return: inverse sf
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        return self._dist.ppf(1 - q)
    
    def stats(self, moments='mv'):
        """
        Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

        :param moments: moments to be returned.
        :return: moments.
        :rtype: tuple
        """
        t_ = []
        if 'm' in moments:
            t_.append(self.mean())
        if 'v' in moments:
            t_.append(self.var())
        if 's' in moments:
            t_.append(self.moment(3) / self.moment(2) ** (3 / 2))
        if 'k' in moments:
            t_.append(self.moment(4) / self.moment(2) ** 2 - 3)
        try:
            assert len(t_) != 0, logger.error("moments argument is not composed of letters 'mvsk'")
        except AssertionError as msg:
            print(msg)

        return tuple(t_)

    def median(self):
        """
        Median of the distribution.

        :return: median
        :rtype: ``numpy.float64``
        """
        return self.ppf(0.5)

    def skewness(self):
        """
        Skewness (third standardized moment).

        :return: skewness.
        :rtype: ``float``
        """
        return self.stats(moments='s')
    
    def kurtosis(self):
        """
        Excess kurtosis.
        
        :return: Excess kurtosis.
        :rtype: ``float``
        """
        return self.stats(moments='k')

    def lev(self, v):
        """
        Limited expected value, i.e. expected value of the function min(x, v).

        :param v: values with respect to the minimum.
        :type v: ``numpy.float`` or ``numpy.ndarray``
        :return: expected value of the minimum function.
        :rtype: ``numpy.float`` or ``numpy.ndarray``
        """

        v = hf.arg_type_handler(v)
        output = v.copy().astype(np.float64)

        if 1 >= self.rate:
            # return all infinity
            return output + float('inf')

        output[v == np.inf] = self.mean()
        filter_ = (v > 0) * (v < np.inf)
        if np.any(filter_):

            factor1 = stats.gamma.cdf(
                np.log(v[filter_]) * (self.rate - 1),
                a = self.a,
                loc = 0,
                scale = 1
                )
            factor2 = stats.gamma.sf(
                np.log(v[filter_]) * self.rate,
                a = self.a,
                loc = 0,
                scale = 1
                )
            inf_lim = v[filter_] if np.isfinite(v[filter_]) else np.zeros(np.sum(filter_))
            output[filter_] = (1.0 - self.scale)**(-self.a) * factor1 + inf_lim * factor2

        return output

    def censored_moment(self, n, d, c):
        """
        Non-central moment of order n of the transformed random variable min(max(x - d, 0), c).
        When n = 1 it is the so-called stop loss transformation function.
        
        :param d: deductible, or attachment point.
        :type d: ``int``, ``float``
        :param c: cover, deductible + cover is the detachment point.
        :type c: ``int``, ``float``
        :param n: moment order.
        :type n: ``int``
        :return: raw moment of order n.
        :rtype: ``float``
        """
        return hf.censored_moment(n=n, d=d, c=c, dist=self)
    
    def partial_moment(self, n, low, up):
        """
        Non-central partial moment of order n.

        :param n: moment order.
        :type n: ``int``
        :param low: lower limit of the partial moment.
        :type low: ``int``, ``float``
        :param up: upper limit of the partial moment.
        :type up: ``int``, ``float``
        :return: raw partial moment of order n.
        :rtype: ``float`` 
        """
        return hf.partial_moment(n=n, low=low, up=up, dist=self) 
    
    def truncated_moment(self, n, low, up):
        """
        Non-central truncated moment of order n.

        :param n: moment order.
        :type n: ``int``
        :param low: lower truncation point.
        :type low: ``int``, ``float``
        :param up: upper truncation point.
        :type up: ``int``, ``float``
        :return: raw truncated moment of order n.
        :rtype: ``float`` 
        """
        adj = self.cdf(up) - self.cdf(low)
        return hf.partial_moment(n=n, low=low, up=up, dist=self) / adj


# Uniform
class Uniform(Beta):
    """
    Uniform continuous distribution over interval [a, b].

    :param a: lower bound.
    :type a: ``float`` or ``int``
    :param b: upper bound.
    :type b: ``float`` or ``int``
    """

    def __init__(self, a=0, b=1):
        Beta.__init__(
            self,
            a = 1,
            b = 1,
            loc = a,
            scale = b
            )
        hf.check_condition(a, b, 'a', logger, '<')
    
    @property
    def a(self):
        return self.__a

    @a.setter
    def a(self, value):
        hf.assert_type_value(value, 'a', logger, (float, int))
        self.__a = value

    @property
    def b(self):
        return self.__b

    @b.setter
    def b(self, value):
        hf.assert_type_value(value, 'b', logger, (float, int))
        self.__b = value

    @staticmethod
    def name():
        return 'uniform'


class Multinomial(_MultDiscreteDistribution):
    """
    Multinomial distribution.
    Wrapper to scipy multinomial distribution (``scipy.stats._multivariate.multinomial``).
    Refer to :py:class:'~__MultDiscreteDistribution' for additional details.

    :param loc: location parameter (default=0), to shift the support of the distribution.
    :type loc: ``int``, optional
    :param seed: Used to set a specific seed (default=np.random.RandomState).
    :type seed: int
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *n* (``int``) --
          Number of trials.
        * *p* (``float``) --
          Probability of a success, parameter of each marginal distribution.

    """

    def __init__(self, loc=0, seed=None, **kwargs):
        _DiscreteDistribution.__init__(self)
        self.n = kwargs['n']
        self.p = kwargs['p']
        self.loc = loc
        self.seed = seed
        
        self.marginals = [Binom(n=self.n, p=p_i) for p_i in self.p]
        
    @property
    def seed(self):
        return self.__seed
    
    @seed.setter
    def seed(self, value):
        if value is None:
            value = np.random.randint(1, 1001)
        
        hf.assert_type_value(value, 'seed', logger, (float,int))
        value = int(value)
        self.__seed = value
 

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        hf.assert_type_value(value, 'n', logger, (float, int), lower_bound=1, lower_close=True)
        value = int(value)
        self.__n = value

    @property
    def p(self):
        return self.__p
    
    @p.setter
    def p(self, value):

        for element in value:
            hf.assert_type_value(element, 'p', logger, (float, np.floating), lower_bound=0, upper_bound=1, lower_close=True, upper_close=True)
                
        value = np.array(value)
        self.__p = value

    @property
    def loc(self):
        return self.__loc

    @loc.setter
    def loc(self, value):
        hf.assert_type_value(value, 'loc', logger, (float, int))
        value = int(value)
        self.__loc = value

    @property
    def _dist(self):
        return stats.multinomial(n=self.n, p=self.p, seed=self.seed)

    @staticmethod
    def name():
        return 'multinomial'
    
    @staticmethod
    def category():
        return {'frequency'}
    
    
    def cov(self):
       """
        Covariance Matrix of a Multinomial Distribution.
        :return: Covariance Matrix.
        :rtype: ``float``
        """

       return stats.multinomial.cov(n=self.n, p=self.p)

    
    def var(self):
       """
        Variances of a Multinomial Distribution.

        :return: Array of Variances.
        :rtype: numpy.ndarray
        """
       return np.diag(stats.multinomial.cov(n=self.n, p=self.p))
   
    
    def entropy(self):
        """
        (Differential) entropy of the Multinomial distribution

        :return: entropy
        :rtype: ``numpy.ndarray``
        """
        return stats.multinomial.entropy(n=self.n, p=self.p)
    
    
    def pmf(self, x):
        """
        Probability mass function of the Multinomial distribution

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        if sum(x) != self.n:
         raise ValueError("n != sum(x), i.e. one is wrong")
        
        return stats.multinomial.pmf(n=self.n, p=self.p, x=x)
    
    def logpmf(self, x):
        """
        Natural logarithm of the probability mass function of the Multinomial distribution

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``
        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``

        """
        if sum(x) != self.n:
         raise ValueError("n != sum(x), i.e. one is wrong")
            
        return stats.multinomial.logpmf(n=self.n, p=self.p, x=x)



class Dirichlet_Multinomial(_MultDiscreteDistribution):
    """
    Dirichlet Multinomial distribution.
    Wrapper to scipy dirichlet multinomial distribution (``scipy.stats.dirichlet_multinomial``).
    Refer to :py:class:'~__MultDiscreteDistribution' for additional details.


    :param seed: Used to set a specific seed (default=np.random.RandomState).
    :type seed: int
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *alpha* ( ``int`` or ``numpy.ndarray``) --
          Concentration parameters.           
        * *n* (``int``) --
          Number of trials.
    """
    
    def __init__(self, seed=None, **kwargs):
        _DiscreteDistribution.__init__(self)
        self.n = kwargs['n']
        self.alpha = kwargs['alpha']
        self.seed = seed
               
    @property
    def seed(self):
        return self.__seed
    
    @seed.setter
    def seed(self, value):
        if value is None:
            value = np.random.randint(1, 1001)
        
        hf.assert_type_value(value, 'seed', logger, (float,int))
        value = int(value)
        self.__seed = value
 
    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        hf.assert_type_value(value, 'n', logger, (float, int), lower_bound=1, lower_close=True)
        value = int(value)
        self.__n = value

    @property
    def alpha(self):
        return self.__alpha
    
    @alpha.setter
    def alpha(self, value):
        for element in value:
            hf.assert_type_value(element, 'alpha', logger, (float, int))                
        value = np.array(value)
        self.__alpha = value

    @property
    def _dist(self):
        return stats.dirichlet_multinomial(n=self.n, alpha=self.alpha, seed=self.seed)

    @staticmethod
    def name():
        return 'dirichlet multinomial'
    
    @staticmethod
    def category():
        return {'frequency'}
        
    def cov(self):
       """
        Covariance Matrix of a Dirichlet Multinomial Distribution.
        
        :return: Covariance Matrix.
        :rtype: ``float``
        """

       return stats.dirichlet_multinomial.cov(n=self.n, alpha=self.alpha)
   
    
    def var(self):
       """
        Variances of a Dirichlet Multinomial Distribution.
        
        :return: Array of Variances.
        :rtype: numpy.ndarray
        """
        
       return np.diag(stats.dirichlet_multinomial.cov(n=self.n, alpha=self.alpha))
 
    def pmf(self, x):
        """
        Probability mass function of the Dirichlet Multinomial Distribution.

        :param x: quantile where probability mass function is evaluated.
        :type x: ``int``

        :return: probability mass function.
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        
        if sum(x) != self.n:
         raise ValueError("n != sum(x), i.e. one is wrong")    
        return stats.dirichlet_multinomial.pmf(n=self.n, alpha=self.alpha, x=x)
    
    def logpmf(self, x):
        """
        Natural logarithm of the probability mass function of the Dirichlet Multinomial Distribution.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``
        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """
        
        if sum(x) != self.n:
         raise ValueError("n != sum(x), i.e. one is wrong")       
        return stats.dirichlet_multinomial.logpmf(n=self.n, alpha=self.alpha, x=x)


    def mean(self):
        """
        Mean of the Dirichlet Multinomial Distribution.

        :param x: quantile where the (natural) probability mass function logarithm is evaluated.
        :type x: ``int``
        :return: natural logarithm of the probability mass function
        :rtype: ``numpy.float64`` or ``numpy.ndarray``
        """        
        return stats.dirichlet_multinomial.mean(n=self.n, alpha=self.alpha)


    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """
        
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)
        hf.assert_type_value(size, 'size', logger, (float, int), lower_bound=1)
        size = int(size)       
        alpha = self.alpha
        n = self.n    
        if isinstance(alpha, np.ndarray) and len(alpha.shape) == 1:
            alpha = np.tile(alpha, (size, 1))
            n = np.full(size, n)                
        G = np.random.gamma(shape=alpha, scale=1.0)
        prob = G / np.sum(G, axis=1, keepdims=True)        
        ridx = np.sum(G, axis=1) == 0        
        if np.any(ridx):
            for i in np.where(ridx)[0]:
                prob[i, :] = np.random.multinomial(1, alpha[i, :] / np.sum(alpha[i, :]), n=1).flatten()                
        rdm = np.array([np.random.multinomial(n[i], prob[i, :]) for i in range(size)])      
        return rdm


class NegMultinom(_MultDiscreteDistribution):
    """
    Negative Multinomial distribution.

    :param loc: location parameter (default=0), to shift the support of the distribution.
    :type loc: ``int``, optional
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * *b* (``int``) --
          dispersion parameter of the negative binomial distribution.
        * *p* (``float``) --
          Probability parameter of the negative binomial distribution.

    """

    def __init__(self, loc=0, **kwargs):
        _DiscreteDistribution.__init__(self)
        self.beta = kwargs['beta']
        self.p = kwargs['p']
        self.loc = loc
        
    @property
    def beta(self):
        return self.__beta

    @beta.setter
    def beta(self, value):
        hf.assert_type_value(value, 'beta', logger, (float, int))
        self.__beta = value

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):

        for element in value:
            hf.assert_type_value(element, 'p', logger, (float, np.floating), lower_bound=0, upper_bound=1, lower_close=True, upper_close=True)
        value = np.array(value)
        
        if sum(value) >= 1:
         raise ValueError("success probabilities must not be greater than 1.")
        
        self.__p = value

    @staticmethod
    def name():
        return 'negative multinomial'

    @staticmethod
    def category():
        return {'frequency'}
    
    
    def pmf(self, x):

        return mt.exp(self.logpmf(x))
    
    def logpmf(self, x):   
        m = np.sum(x, axis=1)
        d = x.shape[1]
        logl = gammaln(self.beta + m) - gammaln(self.beta) - np.sum(gammaln(x + 1), axis=1)
        logl += np.sum(x * np.log(self.p)) + self.beta * np.log(1-np.sum(self.p))
    
        return logl 

    def cov(self, random_state=None):
       """
        Covariance Matrix of a Negative Multinomial Distribution.
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: Covariance Matrix.
        :rtype: ``float``
        """
        
       random_state = hf.handle_random_state(random_state, logger)
    
       return np.cov(self.rvs(size=10000, random_state=random_state).T)
   
    
    def var(self, random_state=None):
       """
        Variances of a Negative Multinomial Distribution.
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: Array of Variances.
        :rtype: numpy.ndarray
        """
       random_state = hf.handle_random_state(random_state, logger)
        
       return np.diag(self.cov(random_state=random_state))
   
    
    def rvs(self, size=1, random_state=None):
        """
        Random variates generator function.

        :param size: random variates sample size (default is 1).
        :type size: ``int``, optional
        :param random_state: random state for the random number generator.
        :type random_state: ``int``, optional
        :return: random variates.
        :rtype: ``numpy.int`` or ``numpy.ndarray``

        """        
        prob = self.p
        beta = self.beta
        
        random_state = hf.handle_random_state(random_state, logger)
        np.random.seed(random_state)
             
        prob = np.tile(prob, (size, 1))      
        k = prob.shape[1]
        probbeta = 1 - np.sum(prob, axis=1)        
        prob = np.hstack((prob, probbeta[:, np.newaxis]))
        scale = 1 / probbeta - 1
        
        G = stats.gamma.rvs(a=beta, scale=scale, size=size)
        lambda_ = prob[:, :k] * (G / (1 - probbeta))[:, np.newaxis]
        
        return np.array([stats.poisson.rvs(mu=l) for l in lambda_])
