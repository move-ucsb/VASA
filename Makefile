make: 
	sphinx-autobuild doc doc/_build/html

uml:
	pyreverse VASA -o png -d UML

build:
	python -m build
	pyhton -m twine upload --repository pypi dist/*