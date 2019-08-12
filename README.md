# lorenz_96_sde

This is a public repository for the model and data analysis scripts from the "On the numerical integration of the Lorenz-96 model, with additive noise, for benchmark twin experiments" project.

This contains the executable scripts for the experiments used in the above manuscript.  Experiments are built in a functional framework so they are easily imported into workers in the Ipyparallel
framework for naive parallelism across parameter regimes, see e.g., https://ipyparallel.readthedocs.io/en/latest/.  Scripts denoted parallel_&lowast;.py are run scripts for the parallel experiments.
For description of each experiment, see the comments under the exp function defined in the script.  Auxilliary functions are defined above this.  

## Getting Started

The following scripts are responsible for generating the experimental data. The initial conditions and truth-twin in DA experiments are run offline with the generate_tay_sde_obs.py, and appropriate
data should be generated this way before running other scripts which depend on this data. 

1. The l96.py script has the general formulation of the strong order 2.0 Talyor integration scheme, for arbitrary order p of truncation of the Fourier series. This also contains other general
utility functions for the L96 model.  The function l96_2tay_sde here is the general model integration code for practical uses, and differs in implementation in other scripts due to the 
need to re-use the same Brownian motion realizations in the benchmarks.

2. The generate_tay_sde_obs.py script is used to generate the long climatological simulation in the L96-s model, used for the initial conditions in various experiments, 
as well as to create the truth-twin in DA experiments, all offline. See parallel_tay_obs.py for generating the data across parameter regimes in parallel.

3. The weak_strong_convergence.py script is used to evaluate the discretization error of the different integration methods in the weak and strong sense over a single initial condition/ Brownian motion
realization.  This uses modified versions of the integration schemes to ensure that all schemes are discretizing the ideal (finely discretized) sample path with respect to the same Brownian motion
realization. See the script parallel_weak_strong.py for generating the data across parameter regimes in parallel. 

4. The ensemble_statistics_analysis.py script generates an ensemble of realizations from a single initial condition with respect to different Brownian motion realizations and computes the ensemble mean
 and the ensemble spread with respect to different integration schemes.  See the script parallel_ensemble_analysis.py for generating the data across parameter regimes in parallel.

5. The vary_ensemble_integration.py script generates the EnKF analysis statistics with respect to a single observation process realization, and realization of the other associated noise processes, while
varying the numerical method of generating the ensemble-based forecast. See the script parallel_ensemble_bias.py for generating the data across parameter regimes in parallel.


## Processing and plotting data

The directory "Data" has additional scripts that are used to process data and produce publication figures.  The plotting scripts are not well documented, but use standard
Matplotlib conventions with figures and axes classes.  If there are questions on using them, they can be directed to the corresponding author.


## License

This project is licensed under the MIT License - see the [License.md](https://github.com/cgrudz/lorenz_96_sde/blob/master/LICENSE.md) file for details

