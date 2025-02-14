
SITE_LINK = 'https://gem-analytics.github.io/gemact/'
DCEILING = 5  # lossaggregation
PROB_TOLERANCE = (1-0.9999) # lossmodel
LOSS_AGGREGATION_METHOD = {'aep', 'mc'}  # lossaggregation
SEV_DISCRETIZATION_METHOD = {'localmoments', 'massdispersal', 'lowerdiscretization', 'upperdiscretization'}  # lossmodel
AGGREGATE_LOSS_APPROX_METHOD = {'fft', 'mc', 'recursion', 'qmc'}  # lossmodel
POLICY_LAYER_BASIS = {'regular', 'drop-down', 'stretch-down'} # lossmodel
QMC_SEQUENCE_METHOD = {'halton', 'sobol', 'lhs'} # lossmodel
POLICY_LAYER_CATEGORY = {'xlrs', 'xl/sl'} # lossmodel
RESERVING_METHOD = {'fisher_lange', 'crm'} # lossreserve
# dictionary of copulas
COP_DICT = {
    'clayton': 'copulas.ClaytonCopula',
    'frank': 'copulas.FrankCopula',
    'gumbel': 'copulas.GumbelCopula',
    'gaussian': 'copulas.GaussCopula',
    'joe': 'copulas.JoeCopula',
    'ali-mikhail-haq': 'copulas.AliMikhailHaqCopula',
    'tstudent': 'copulas.TCopula',
    'independent': 'copulas.IndependenceCopula',
    'frechet–hoeffding-lower': 'copulas.FHLowerCopula',
    'frechet–hoeffding-upper': 'copulas.FHUpperCopula'
    }
# dictionary of distributionss
DIST_DICT = {
    'binom': 'distributions.Binom',
    'geom': 'distributions.Geom',
    'nbinom': 'distributions.NegBinom',
    'poisson': 'distributions.Poisson',
    'zmbinom': 'distributions.ZMBinom',
    'zmgeom': 'distributions.ZMGeom',
    'zmlogser': 'distributions.ZMLogser',
    'zmnbinom': 'distributions.ZMNegBinom',
    'zmpoisson': 'distributions.ZMPoisson',
    'ztgeom': 'distributions.ZTGeom',
    'ztbinom': 'distributions.ZTBinom',
    'ztnbinom': 'distributions.ZTNegBinom',
    'ztpoisson': 'distributions.ZTPoisson',
    'beta': 'distributions.Beta',
    'burr12': 'distributions.Burr12',
    'dagum': 'distributions.Dagum',
    'exponential': 'distributions.Exponential',
    'fisk': 'distributions.Fisk',
    'gamma': 'distributions.Gamma',
    'genbeta': 'distributions.GenBeta',
    'genpareto': 'distributions.GenPareto',  
    'invgamma': 'distributions.InvGamma',
    'invgauss': 'distributions.InvGauss',
    'invweibull': 'distributions.InvWeibull',
    'lognormal': 'distributions.Lognormal',
    'loggamma': 'distributions.LogGamma',
    'weibull': 'distributions.Weibull',
    'pwl': 'distributions.PWL',
    'pwc': 'distributions.PWC',
    'paralogistic': 'distributions.Paralogistic',
    'invparalogistic': 'distributions.InvParalogistic',
    'pareto2': 'distributions.Pareto2',
    'pareto1': 'distributions.Pareto1',
    'uniform': 'distributions.Uniform',
    'multinomial': 'distributions.Multinomial',
    'dirichlet multinomial' : 'distributions.Dirichlet_Multinomial',
    'negative multinomial' : 'distributions.NegMultinom'
}
