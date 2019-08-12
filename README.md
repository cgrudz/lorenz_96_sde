# lorenz_96_sde

This is a public repository for the model and data analysis scripts from the "On the numerical integration of the Lorenz-96 model, with additive noise, for benchmark twin experiments" project.

This contains the executable scripts for the experiments used in the above manuscript.  Experiments are built in a functional framework so they are easily imported into workers in the Ipyparallel
framework for naive parallelism across parameter regimes, see e.g., https://ipyparallel.readthedocs.io/en/latest/.  Scripts denoted parallel_&lowast;.py are run scripts for the parallel experiments.  

## Getting Started


## License

This project is licensed under the MIT License - see the [License.md](https://github.com/cgrudz/lorenz_96_sde/blob/master/LICENSE.md) file for details

