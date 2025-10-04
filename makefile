PY=python
DATA_DIR=src/data/

SUBMIT_FILE?=submission.csv
SUBMIT_MSG?="Submission"

MODE?=ensemble # [ensemble, m1, m2]
EXECUTION?=both # [cv, train, both]
SEED?=42

M2_M?=300.0
M2_M_GRID?="1.0,50.0,100.0,150.0,200.0,300.0,400.0,500.0,600.0,700.0"
FOLDS?=5

M1_MODELS?="rf, et, xgb, cat" # [rf, et, hgb, gbr, enet, svr, knn, xgb, cat]
M2_MODELS?="rf, et, xgb, cat" # [rf, et, hgb, gbr, enet, svr, knn, xgb, cat]
INNER_SPLITS?=3
INNER_ITER?=20

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
		--m2-blend-bag-boost \
		--show-importance

m_search:
	$(PY) -m src.modelling.train \
		--seed $(SEED) \
		--m2-m $(M2_M) \
		--m2-m-grid $(M2_M_GRID) \
		--folds 10 \
		--mode ensemble \
		--execution both \
		--m1-models rf \
		--m2-models rf \
		--inner-splits 3 \
		--inner-iter 0 \
		--show-importance
