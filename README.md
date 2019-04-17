# StarPopSim - simulating stellar populations for mock images


## What is StarPopSim?
StarPopSim is a simulation code written in pure Python that can generate a single or multiple stellar populations in various spatial distributions to mimic real astronomical objects as closely as possible. 
The aim is to provide the necessary quantities for other code packages that can simulate the workings of a telescope optical train and instruments. (This feature could also be crudely implemented natively at a later stage.)
The basic input parameters are, among others, the number of stars (or mass in stars), the age of the population(s), the metallicity for each population and the distance to your object.

Currently, with version 1.0 (April 2019), stellar (globular) clusters are the best implemented feature, with the possibility to scale to elliptical galaxies. Other galaxy types are hopefully implemented at a future stage.

At this stage, the implemented instrument simulator is SimCADO. This code simulates the workings of MICADO at ELT, more on that can be found in [Leschinski et al. (2016)](https://arxiv.org/pdf/1609.01480v1.pdf).


### Reference Material

* Detailed documentation on what StarPopSim can do is found here: [master-thesis-WIP](no-link-yet)

* If you want to contribute to better documentation, please send an e-mail to the adress below


## Getting started
As of version 1.0 (April 2019), the way to get this code is to either download it or make a fork on GitHub. To use it, one can use the command line or a Python environment of yourchoosing to execute the functions.

**To keep StarPopSim updated**, grabbing a copy from the master branch should always provide you with the latest working version.

**StarPopSim has only been tested in Python 3.6**. 
Using older versions could result in unexpected errors, although any Python 3 version is expected to work.

**Package dependencies:** Numpy, Scipy, matplotlib, astropy (for .fits functionality), SimCADO (for imaging functionality)

**Note:** making the astronomical objects in StarPopSim is in principle fully independent from SimCADO. So if different instrument simulation software is going to be used instead, a SimCADO install is not necessary.


### Running simulations

To create a basic source object, the following lines of python are sufficient (assuming the modules are in the current working directory):

    >>> import objectgenerator as obg
	>>> astobj = obg.AstObject(M_tot_init=10**5, age=[10], metal=[0.0014], distance=10**3)

This will create a cluster of stars that has a total initial mass of 1e5 solar masses with one population of stars with an age of 1e10 years and a metallicity of Z=0.0014 at a distance of 1e3 parsecs.
Many more options are available. For a more detailed description I refer the reader to the reference material listed above.
Alternatively, the same can be achieved directly from the command line with:

	>>> python3 constructor.py -M 10**5 -ages 10 -z 0.0014 -D 10**3
	
To find out how to create the object you want (via the command line), there is an interactive object builder that will give out the object as well as the command line command that will do exactly that

	>>> python3 constructor.py -inter

**Note:** this feature might not include all the latest upgrades to the core functionality.

The AstObject can be saved to a Pickles file by simply calling:

	>>> astobj.SaveTo(filename)
	
And it is recovered with:

	>>> astobj = obg.AstObject.LoadFrom(filename)

### Imaging

Similarly, to create images of the objects, we import the image generator module and make a SimCADO source object out of our *AstObject*:

	>>> import imagegenerator as img
	>>> src = img.MakeSource(astobj, filter='V')
	
Where the first parameter needed is the object you have just created (or previously saved to disc), and the second one is the imaging filter that is going to be used later.
It is recommended the same filter be used here as in the imaging below. An image is then made by calling:

	>>> image = img.MakeImage(src, exposure=60, NDIT=1, view='wide', chip='centre', filter='V', ao_mode='scao', 
				  filename='image_default_save', internals=None, return_int=False)

where the first parameter is the just created source object and the other parameters give most of the relevant (basic) functionality of SimCADO.
For more documentation on SimCADO see [simcado.readthedocs.io](https://simcado.readthedocs.io/en/latest/index.html) (it can do a lot more!).
Alternatively, there is a one-liner that achieves (almost) the same (might be updated to do all the same things in the future):

	>>> python3 imager.py -astobj default_object_file_name -exp 60 -ndit 1 -filter V -fov wide 
				-chip centre -ao scao -save image_default_save
	
Or execute the interactive function to get some guidance in all the available options for simulation:

	>>> python3 imager.py -inter
	
### Showing images

StarPopSim comes with a handy fits-file-handling module that does all the basic fits manipulations as well as some more involved ones. 
The two most notable are saving (to png) and plotting a fits image:

	>>> import fitshandler as fh
	>>> fh.SaveFitsPlot(filename, index=0, colours='gray', grid=True)
	>>> fh.PlotFits(filename, index=0, colours='gray', grid=True)
	
There is another module for visualisation, that can make various plots of the AstObject made above. To give two examples:

	>>> import visualizer as vis
	>>> vis.Objects2D(objects, title='Scatter', xlabel='x', ylabel='y', axes='xy', 
					  colour='blue', T_eff=None, dark_theme=True)

This will make a two-dimensional plot of the object projected on the x-y plane. The colour can be set to 'temperature', if T_eff is specified for each star. The dark theme is just a preset with some nice plotting parameters.
The following function will make an HR-diagram of the stellar population(s):

	>>> vis.HRD(T_eff, log_Lum, title='HRD', xlabel='Temperature (K)', ylabel=r'Luminosity log($L/L_\odot$)', 
				colour='temperature', dark_theme=True, mask=None)

Here, a boolean mask can be specified to take out specific stars (since they might fall far outside of the wanted plotting area).


## Use of units

Several quantities are used throughout the code. Where no units are specified, the following applies. Masses are always in solar mass, luminosity is in units of solar luminosity. Distances are in parsec, if not stated otherwise. Temperatures are in Kelvin.
The age of a population is in years, but if a number below 12 is given, it is assumed to be in log10(years).


## Bugs and Issues

Despite all the testing, I am certain that there are still bugs in this code, or will be created in future versions. 

If you happen to come across any bugs or issues, *please* contact me. Only known bugs can be resolved.
This can be done through opening an issue on the StarPopSim GitHub page: [LucIJspeert/StarPopSim/issues](https://github.com/LucIJspeert/StarPopSim/issues), or by contacting me directly (see below).

If you are (going to be) working on new or improved features, I would love to hear from you and see if it can be implemented in the source code.


## Contact

For questions and suggestions, please contact:

* l.ijspeert(at)student.ru.nl

or, in case the above adress is no longer in use:

* lucijspeert(at)gmail.com

**Developer (Nijmegen):** Luc IJspeert

