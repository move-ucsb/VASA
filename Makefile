make: 
	sphinx-autobuild doc doc/_build/html

uml:
<<<<<<< HEAD
<<<<<<< HEAD
	pyreverse VASA -o png -d UML

build:
	python3 -m build
	python3 -m twine upload --repository pypi dist/*
=======
	pyreverse VASA -o png -d UML
>>>>>>> 32dc93af2d375f32c56053c710c69184224c9a08
=======
	pyreverse VASA -o png -d UML
>>>>>>> 32dc93af2d375f32c56053c710c69184224c9a08
