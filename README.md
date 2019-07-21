# project
- N. Varghese, A. Vitek, Z. Zhao

## Before running the code
The training data is large (~10GB uncompressed). It can be downloaded with the following instructions. Alternatively, the iPython notebooks contain saved states with the results.
1. `kaggle competitions download -c petfinder-adoption-prediction`
2. Ensure the training data is extracted in `./all`
3. Download https://github.com/git-lfs/git-lfs/releases/download/v2.7.1/git-lfs-darwin-amd64-v2.7.1.tar.gz
4. `git lfs install`
5. `git lfs pull`

## Contents
- `all` contains a single txt file, which will be populated once `git lfs pull` is performed. Also, it should contain the downloaded training data
- `attempts` contains the approaches described in the project report. Three are iPython notebooks, one is a regular script
- `checkpoints` contains the best models we obtained
- `data_exploration` contains iPython notebooks containing the data analytics we performed, the the Decision Tree discovery in the traditional machine learning pre-trials we made for the project
