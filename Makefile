.PHONY: reformat
reformat:
	black */*.py

.PHONY: test
test:
	cd test; pytest test*.py
