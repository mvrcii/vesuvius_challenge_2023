# Modify .bashrc
echo "export CC=/usr/bin/gcc" >> ~/.bashrc
echo "export PS1='\\[\\033[38;5;79m\\]\\u@\\h\\[\\033[00m\\]:\\[\\033[38;5;33m\\]\\w\\[\\033[00m\\]$ '" >> ~/.bashrc
echo "alias pp='export PYTHONPATH=\"\$PWD:\$PYTHONPATH\"'" >> ~/.bashrc
echo "alias vsv='cd ~/kaggle1stReimp && pp'" >> ~/.bashrc

# Create local config file
touch conf_local.py
echo -e 'import os\nwork_dir = os.path.join("")' > conf_local.py

# Git
git config --global credential.helper cache
git config --global credential.helper 'cache --timeout=1209600'

# Install requirements
pip install -r requirements.txt
sudo apt-get install libgl1-mesa-glx

# Download Fragments
python ./util/batch_download_frags_multilayer.py

