
make: 
	sphinx-autobuild doc doc/_build/html

uml:
	pyreverse -o png -p Pyreverse pylint/pyreverse/