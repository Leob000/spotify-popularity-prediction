PY=python
DATA_DIR=src/data/

SUBMIT_FILE?=submission.csv
SUBMIT_MSG?="Submission"

MODE?=ensemble # [ensemble, m1, m2]
EXECUTION?=both # [cv, train, both]
SEED?=42

M2_M?=150.0
FOLDS?=5

M1_MODELS?="rf" # [rf, et, hgb, gbr, enet, svr, knn, xgb, cat]
M2_MODELS?="rf" # [rf, et, hgb, gbr, enet, svr, knn, xgb, cat]
# INNER_SPLITS?=4
INNER_SPLITS?=3
INNER_ITER?=0

submit:
	kaggle competitions submit -c spotify-predire-la-popularite-dun-titre -f $(DATA_DIR)$(SUBMIT_FILE) -m $(SUBMIT_MSG)

train:
	$(PY) -m src.modelling.train \
		--seed $(SEED) \
		--m2-m $(M2_M) \
		--folds $(FOLDS) \
		--mode $(MODE) \
		--execution $(EXECUTION) \
		--m1-models $(M1_MODELS) \
		--m2-models $(M2_MODELS) \
		--inner-splits $(INNER_SPLITS) \
		--inner-iter $(INNER_ITER) \
		--show-importance
