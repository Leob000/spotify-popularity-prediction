PY=python
DATA_DIR=src/data/

SUBMIT_FILE?=submission.csv
SUBMIT_MSG?="Submission"

MODE?=m2 # [ensemble, m1, m2]
SEED?=42

M2_M?=150.0
FOLDS?=5

submit:
	kaggle competitions submit -c spotify-predire-la-popularite-dun-titre -f $(DATA_DIR)$(SUBMIT_FILE) -m $(SUBMIT_MSG)

train:
	$(PY) -m src.modelling.train \
		--seed $(SEED) \
		--m2_m $(M2_M) \
		--folds $(FOLDS) \
		--mode $(MODE) \
		--show_importance
