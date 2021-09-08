make: 
	sphinx-autobuild doc doc/_build/html

uml:
	pyreverse VASA -o png -d UML

build:
<<<<<<< HEAD
<<<<<<< HEAD
	python -m build
	pyhton -m twine upload --repository pypi dist/*
=======
=======
	rm -rf dist/*
>>>>>>> ff86c210a7c4af50f463113a1d14c987eb1154c8
	python3 -m build
	python3 -m twine upload --repository testpypi dist/* --verbose
>>>>>>> 21f44e38e3123064ecabd9bdadb5350b45e4f847
