# NERSC setup

module load conda

conda activate /global/common/software/nintern/gzquse/

# local conda env
pip install --user -r .\requirements.txt

dwave auth login
dwave auth get
dwave setup --auth