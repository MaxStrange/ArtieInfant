.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
DATE = `date +'%d_%m_%y_%H_%M_%S'`
PROFILE = default
PROJECT_NAME = mlpvad
PYTHON_INTERPRETER = python3
PIP = pip3
THEANO_FLAGS = device=gpu,floatX=float32
HD_DEV = blah

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Define a function for printing in color.
define colorecho
	@tput setaf 1
	@echo $1
	@tput sgr0
endef

## Install Python Dependencies
requirements: test_environment
	@echo "Ensuring all requirements are installed..."
	@$(PIP) install pipreqs
	@$(PIP) install h5py
	@pipreqs --force .
	@$(PIP) install --upgrade -r requirements.txt
	@if [ ! -d ~/.keras ]; then mkdir ~/.keras; echo "{\n    \"image_data_format\": \"channels_last\",\n    \"epsilon\": 1e-07,\n    \"floatx\": \"float32\",\n    \"backend\": \"theano\"\n}" > ~/.keras/keras.json; fi

## Make Dataset
data:
ifeq (,$(HD_DEV))
ifeq (,$(wildcard data/raw/.makeskip))
	$(call colorecho, ":::::: FAIL ::::::")
	$(call colorecho, "You must modify the Makefile's HD variable to be the path to the device.")
	$(call colorecho, "::::::::::::::::::")
endif
else
	$(PYTHON_INTERPRETER) -m src.data.make_dataset $(HD_DEV) $(PROJECT_DIR)/data
	if [ $$? -eq 0 ] ; then touch data/raw/.makeskip ; fi
endif

## Train the model(s)
train:
	$(THEANO_FLAGS) $(PYTHON_INTERPRETER) -m src.models.train_model $(PROJECT_DIR)/models/mlps/$(DATE).hdf5 $(PROJECT_DIR)/data/processed

## Run the model on the test split
test:
	$(THEANO_FLAGS) $(PYTHON_INTERPRETER) -m src.models.predict_model $(PROJECT_DIR)/models/mlps/to_test.hdf5 $(PROJECT_DIR)/data/processed

## Run the visualization script to view real time updates
visualize:
	$(PYTHON_INTERPRETER) -m src.visualization.visualizetraining

## Delete all compiled Python files
clean:
	find . -name "*.pyc" -exec rm {} \;

## Lint using flake8
lint:
	flake8 --exclude=lib/,bin/,docs/conf.py .

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Initial setup of the directory structure and python environment
environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@echo ">>> Need to install virtualenv and virtualenvwrapper system-wide, so enter password."
	@sudo pip install virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	@echo ">>> Creating directory structure..."
endif
	@mkdir -p $(PROJECT_DIR)/data/external
	@mkdir -p $(PROJECT_DIR)/data/interim
	@mkdir -p $(PROJECT_DIR)/data/processed
	@mkdir -p $(PROJECT_DIR)/data/raw
	@sudo apt-get install ffmpeg

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := show-help

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
.PHONY: show-help
show-help:
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
