# for documentation
start:
	cd doc; mkdocs serve -a localhost:8080

build:
	cd doc; mkdocs build --clean