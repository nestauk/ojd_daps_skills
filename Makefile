SHELL := /bin/bash

# Detect how to open things depending on our OS
OS = $(shell uname -s)
ifeq ($(OS),Linux)
	OPEN=xdg-open
else
	OPEN=open
endif

# Export all environment variables
export

# Import config variables
include .cookiecutter/config

.PHONY: install
## Install a project: create conda env; install local package; setup git hooks
install: .cookiecutter/state/conda-create .cookiecutter/state/setup-git
	@direnv reload  # Now the conda env exists, reload to activate it


.PHONY: check-bucket-path
check-bucket-path:
	@test ${S3_INPUT_PATH} || (echo 'Please set a S3_INPUT_PATH environment variable (e.g. in .envrc) documenting the S3 path to your inputs/ - e.g. s3://nesta-ds-projects/your-mission/project-name' && exit 1)

.PHONY: inputs-pull
## Pull `inputs/` from S3
inputs-pull: check-bucket-path
	aws s3 sync ${S3_INPUT_PATH} inputs

.PHONY: docs
## Build the API documentation
docs:
	sphinx-apidoc -o docs/api ${REPO_NAME}
	sphinx-build -b docs/ docs/_build

.PHONY: docs-clean
## Clean the built API documentation
docs-clean:
	rm -r docs/source/api
	rm -r docs/_build

.PHONY: docs-open
## Open the docs in the browser
docs-open:
	$(OPEN) docs/_build/index.html

.PHONY: conda-update
## Update the conda-environment based on changes to `environment.yaml`
conda-update:
	conda env update -n ${REPO_NAME} -f environment.yaml
	$(MAKE) -s pip-install
	direnv reload

.PHONY: pip-install
## Install our package and requirements in editable mode (including development dependencies)
pip-install:
	@pip install -e ".[dev]"

.PHONY: conda-remove
## Remove the conda-environment cleanly
conda-remove:
	conda env remove -n ${REPO_NAME}
	rm .cookiecutter/state/conda-create*
	@direnv reload

.PHONY: clean
## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


#################################################################################
# Helper Commands (no need to explicitly document)                              #
#################################################################################

define err
	(echo "$1, check $@.log for more info" && exit 1)
endef

.cookiecutter/state/conda-create:
	@echo -n "Creating environment ${REPO_NAME} and installing all dependencies"
	@(conda env create -q -n ${REPO_NAME} -f environment.yaml\
	  && eval "$$(conda shell.bash activate "${REPO_NAME}")"\
	  && pip install -e ".[dev]")\
	 > $@.log 2>&1\
	 || $(call err,Python environment setup failed)
	@touch $@
	@echo " DONE"

.cookiecutter/state/setup-git:
	@echo -n "Installing and configuring git pre-commit hooks"
	@(eval "$$(conda shell.bash activate "${REPO_NAME}")"\
	 &&pre-commit install --install-hooks)\
	 > $@.log 2>&1\
	 || $(call err,Git pre-commit setup failed)
	@touch $@
	@echo " DONE"


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
