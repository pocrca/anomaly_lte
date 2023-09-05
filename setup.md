# Set up Github account
1. Set up Github user account (if one don't have account)
2. Request for user account to be added to private repository (under 3)
3. Open at web browser: https://github.com/pocrca/anomaly_lte.git  
&nbsp;  

# WSL2 and Ubuntu setup
1. Turn Windows features on or off:
- &#9745; : Virtual Machine Platform 
- &#9745; : Windows Subsystem for Linux  
&nbsp;

2. Check present WSL setup at at Windows command prompt, :
```cmd
wsl -l -v    # check
```
3. Get 'Ubuntu 20.04.5 LTS' from Microsoft Store  
- Open Ubuntu  
&nbsp;  

4. Configure WSL version to be 2:
- Download and install 'wsl_update_x64' if required
```cmd
wsl -l -v

wsl --set-default-version 2
wsl --set-version Ubuntu 20.04 2

wsl -l -v
```  

# VSCode install
1. Get 'Visual Studio Code' from Microsoft Store    
&nbsp;  

# Ubuntu - Virtual enviroment setup  
1. Set up python virtual environment
```bash
sudo apt update

mkdir project
cd project

python3 -m venv venv
sudo apt install python3.8-venv         # if required

```  
2. Add following to '~/.bashrc'
```bash
alias ae='source venv/bin/activate'
```
- at 'project' directory, run 'ae' to activate virtual environment  

# Git: SSH access setup, Clone from Github
1. Set up SSH public key
```bash
ssh-keygen -t 99001 -c "email_used_in_github@example.com"
```
2. Copy and paste SSH public key in user github account: Settings -> (Access) SSH and GPG keys -> New SSH key  
&nbsp;
3. Perform clone at Ubuntu
```bash
git clone git@github.com:pocrca/anomaly_lte.git
```

# Install python libraries
```bash
cd lte_anomaly
pip install -r requirements.txt
```

# VSCode setup
1. Install extensions:
    - WSL
    - Install in WSL:Ubuntu-20.04: 
        * Jupyter
        * Python
2. Open eda.ipynb
3. In VScode: 
```
a. open command palette — Ctrl+Shift+P
b. Look for 'Python: Select Interpreter'
b. In 'Select Interpreter' choose 'Enter interpreter path...' and then 'Find...'
c. Navigate to your 'venv' folder: ~/project
d. In virtual environment folder choose: ~project/venv/bin/python3

```
4. Close VScode. Open VScode again. 
5. Select the kernel from virtual environment (venv)