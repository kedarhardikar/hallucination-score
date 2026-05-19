#view db
python db.py inspect stress
python db.py inspect hotpotqa

#To force a rebuild (e.g. after changing the dataset):
python db.py reset stress

# GPU acceleration
# Set NLI_DEVICE=0 to run the NLI cross-encoder on GPU (default is CPU, device=-1).
# Example:
#   NLI_DEVICE=0 python evaluate.py stress