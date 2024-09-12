.PHONY: reformat
reformat:
	black */*.py

.PHONY: test
test:
	cd test; pytest test*.py

.PHONY: whl
whl:
	python3 m pip install build
	python3 -m build
