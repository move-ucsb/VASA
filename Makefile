make: 
	sphinx-autobuild doc doc/_build/html

uml:
	pyreverse VASA -o png -d UML

build:
	rm -rf dist/*
	python -m build
	python -m twine upload --repository testpypi dist/* --verbose
