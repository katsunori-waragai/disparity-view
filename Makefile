.PHONY: reformat
reformat:
	black */*.py

.PHONY: test
test:
	cd test; pytest test*.py

.PHONY: whl
whl:
	apt install -y python3.8-venv
	python3 -m venv venv
	. venv/bin/activate
	python3 -m pip install build
	python3 -m build
