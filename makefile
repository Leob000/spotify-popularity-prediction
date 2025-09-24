PY=python
DATA_DIR=src/data/

SUBMIT_FILE?=submission.csv
SUBMIT_MSG?="Submission"

submit:
	kaggle competitions submit -c spotify-predire-la-popularite-dun-titre -f $(DATA_DIR)$(SUBMIT_FILE) -m $(SUBMIT_MSG)
