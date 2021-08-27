make: 
	sphinx-autobuild doc doc/_build/html

uml:
	pyreverse VASA -o png -d UML

build:
	python3 -m build
	python3 -m twine upload --repository pypi dist/*
