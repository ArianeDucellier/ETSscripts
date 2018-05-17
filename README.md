# ETSscripts

First, install Git using the instructions here:

[Software Carpentry](https://swcarpentry.github.io/workshop-template/#git)

Clone the GitHub repository:

git clone https://github.com/ArianeDucellier/ETSscripts.git

Then, you need to download and install Anaconda 3:

[Anaconda](https://www.anaconda.com/download/#macos)

You can follow the instructions about Python from here:

[Software Carpentry](https://swcarpentry.github.io/workshop-template/#git)

Then, go to the directory ETSscripts and create an environment:

conda env create -f environment.yml

Finally, you need to add these lines in your .bash_profile

export PYTHONPATH="${PYTHONPATH}:/Users/your_name/your_path/ETSscripts/utils"

export PYTHONPATH="${PYTHONPATH}:/Users/your_name/your_path/ETSscripts/wmtsa"

You should now be able to activate the seismic environment:

source activate seismic

And launch Jupyter Notebook:

jupyter notebook

I think that should work on a Mac. Some things may be different on Windows or Linux.
