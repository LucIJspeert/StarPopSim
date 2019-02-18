####################// Running the code //####################################################################################################

Execute the code from within the main directory, as the code tries to access files in directories within there.

Dependencies: Python 3
	Numpy, Scipy, matplotlib, astropy, SimCADO


####################// Units //####################################################################################################

Masses are always in solar mass. Msol = 1.98855*10^30 kg
Luminosity is in units of solar luminosity. Lsol = 3.828*10^26 W
Distances are in parsec, if not stated otherwise. 


####################// Modules //####################################################################################################

Below follow short descriptions for relevant functions and user inputs

####################/*** constructor.py ***/################################################################################

>>> This module acts as a basic user interface for constructing astronomical objects, as it contains both the option for a
	one-line command to make objects as well as an interactive function ('DynamicConstruct') 
	that leads the user through all the options for making an astronomical object that is as complex as desired.
	-	This interactive function can be called from the command line by using: 
		<python constructor -inter True>
	-	It is advised to use this function for first time use of the code

####################/*** AstObjectClass.py ***/################################################################################

>>> This module contains the class that stores all the information about the astronomical object.
	It can be saved and loaded using Pickle, by calling the built-in functions 'SaveTo' and 'LoadFrom'.

>>> Number vs Mass input
	See text under ObjectGen.py

####################/*** ObjectGen.py ***/################################################################################

>>> Number vs Mass input
	Either a mass or a number of objects can be provided by the user. If both given, the number overrides the mass input.
	-	If a number is given, that exact number of objects will be generated, and the total mass calculated from that.
		The stored mass difference is then the actual total mass minus the estimate for the given number.
	-	If a mass is given, the number of objects to generate is estimated using the integrated IMF.
		The total mass is then updated to the actual sum, and the difference tot the original is stored.

>>> Orientation:
	The objects are given Cartesian coordinates, where the z-axis is assumed to be the observers line of sight.
	-	This means the given distance is a virtual position at z = distance.
	An inclination angle can be defined by the user, to alter the orientation of the object.
	-	This means the given rotation parameter (inclination) is the angle of the original object's x-axis 
		(so perpendicular to z, the l.o.s.) with the new x'-axis (which is then tilted towards positive z).
	For ellipsoidal shapes, a set of relative axes can be defined to change the proportions of the resulting shape.
	-	This means the x,y,z axes are stretched to the given axes ratio.

>>> When using IsocProps, care has to be taken with masses below the lower mass limit in the file (or above the upper).
	Below the lower mass limit, interpolated values will be the lowest available for the specific property.
	The interpolated actual masses will be set to zero for initial masses above the upper limit, and:
	-	logL/Lsun will be set to -9
	-	logTe will be set to 1
	-	magnitudes are set to 20

####################/*** distributions.py ***/################################################################################

>>> Add your own probability density functions (pdf) in the following way:
	1) Normalize the pdf to probability of 1
	2) Integrate the pdf to get the cumulative distribution function (cdf)
	3) Invert the cdf to get the formula for the quantity that you need
		as a function of the density itself
		If inverting is impossible, use numerical inversion (see below)
	4) Pick random numbers between 0 and 1 (recommend using numpy for this)
		and feed them to your inverted cdf

>>> If a desired pdf has a cdf but the inverse of the cdf cannot be found, do the following.
	Implement the cdf(x)
	Calculate the cdf at suitable points for x (often logspace with ~1000 points is good)
	Interpolate the generated values (between 0-1) with as x-points the cdf values and as f-points the x values
	-	This can even be a faster method compared to the analytical inverse cdf.
	-	In general, radial distributions will have to use this method due to their complicated integrated forms.

>>> When a desired radial distribution has to be implemented from a 1 dimensional (x,y,z) distribution, do the following.
	Multiply the 1-d pdf by r^2 and change the variable to r (from 0 to infinity).
	Renormalize.
	Implement in code (following steps 1-4 above).
	-	Or see reference: http://dlibra.umcs.lublin.pl/Content/21054/czas16364_69_2014_1.pdf

>>> Use the right format/nomenclature for implementing the code (mostly for consistency reasons):
	The argument for specifying the amount of generated coorditates must be 'n'
	For radial distributions, the following is used:
		The scaling parameter (or 'core radius') is called 's'
		The maximum (or 'outter') radius is called 'R'
		The function name must end with '_r' (for 'radial'; this is required for the code to recognise the function)

####################/*** conversions.py ***/################################################################################



####################/*** visualizer.py ***/################################################################################



####################/*** fitsHandler.py ***/################################################################################