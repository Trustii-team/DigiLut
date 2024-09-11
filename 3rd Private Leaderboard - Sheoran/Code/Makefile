.PHONY: create_environment install_requirements preprocess_data train predict workflow build_submission

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = CVN_solution
SHELL := /bin/bash

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_environment:
	python -m venv .venv

## Install Python Dependencies
install_dependencies:
	python -m pip install -r requirements.txt
	python -m pip install opencv-python==4.8.0.74

## Preprocess data
preprocess_data:
	jupyter execute notebooks/preprocess_data.ipynb
## Train model(s)
train:
	jupyter execute notebooks/train.ipynb

## Predict
predict:
	jupyter execute notebooks/predict.ipynb

## Project entire pipeline
workflow:
	make create_environment
	source .venv/bin/activate
	make install_dependencies
	make preprocess_data
	make train
	make predict

## Generate jupyter kernel
generate_kernel:
	python -m ipykernel install --user --name=${PROJECT_NAME}_env

## Build submission
build_submission:
	zip submission.zip data/predictions models notebooks requirements.txt .env Makefile README.md

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