### This is a script that runs the pre_generation_entropy_measurement.py script on a running Lambda Cloud instance ###

#!/bin/bash

cd ../

source .env

cd push_t_state_policy_experiments/

# Prompt user for the API key and instance details
read -p "Enter the name of your lambda API key (e.g. niel_lambda_api_key): " user_lambda_api_key_name
USER_LAMBDA_API_KEY=$(eval echo \$$user_lambda_api_key_name)
read -p "Enter the directory location of your private SSH key: " private_ssh_key
read -p "Enter the SSH user (e.g. ubuntu): " remote_ssh_user
read -p "Enter the SSH host/instance address (e.g. 129.146.33.218): " remote_ssh_host
read -p "Enter the name of your huggingface api key in .env file: " huggingface_api_key_name
HUGGINGFACE_API_KEY=$(eval echo \$$huggingface_api_key_name)

# Copy model checkpoint to the remote instance
MODEL_CHECKPOINT_PATH="./pusht_state_100ep.ckpt"

read -p "Would you like to copy the model checkpoint to the remote instance? (y/n): " copy_model
if [[ $copy_model == "y" ]]; then
    echo "Copying model checkpoint to remote instance..."
    scp -i "$private_ssh_key" "$MODEL_CHECKPOINT_PATH" "$remote_ssh_user@$remote_ssh_host:~/$MODEL_CHECKPOINT_PATH"
else
    echo "Skipping model checkpoint copy."
fi

# Copy inference script to the remote instance
MAIN_SCRIPT_PATH="./state_policy_push_t.py"
DATASET_SCRIPT_PATH="./push_t_state_dataset.py"
ENV_SCRIPT_PATH="./push_t_state_env.py"
NETWORK_SCRIPT_PATH="./push_t_state_network.py"
read -p "Would you like to copy the inference scripts to the remote instance? (y/n): " copy_script
if [[ $copy_script == "y" ]]; then
    echo "Copying inference scripts to remote instance..."
    scp -i "$private_ssh_key" "$MAIN_SCRIPT_PATH" "$remote_ssh_user@$remote_ssh_host:~/$MAIN_SCRIPT_PATH"
    scp -i "$private_ssh_key" "$DATASET_SCRIPT_PATH" "$remote_ssh_user@$remote_ssh_host:~/$DATASET_SCRIPT_PATH"
    scp -i "$private_ssh_key" "$ENV_SCRIPT_PATH" "$remote_ssh_user@$remote_ssh_host:~/$ENV_SCRIPT_PATH"
    scp -i "$private_ssh_key" "$NETWORK_SCRIPT_PATH" "$remote_ssh_user@$remote_ssh_host:~/$NETWORK_SCRIPT_PATH"
else
    echo "Skipping script copy."
fi

# Copy dataset to the remote instance
DATASET_PATH="./pusht_cchi_v7_replay"
read -p "Would you like to copy the dataset to the remote instance? (y/n): " copy_dataset
if [[ $copy_dataset == "y" ]]; then
    echo "Copying dataset to remote instance..."
    scp -r -i "$private_ssh_key" "$DATASET_PATH" "$remote_ssh_user@$remote_ssh_host:~/"
else
    echo "Skipping dataset copy."
fi

# Install requirements
read -p "Would you like to install the requirements on the remote instance? (y/n): " install_requirements
if [[ $install_requirements == "y" ]]; then
    echo "Installing requirements on remote instance..."
    ssh -i "$private_ssh_key" "$remote_ssh_user@$remote_ssh_host" "python3 -m venv .venv"
    ssh -i "$private_ssh_key" "$remote_ssh_user@$remote_ssh_host" "source .venv/bin/activate && pip install torch==1.13.1 torchvision==0.14.1 diffusers==0.18.2 scikit-image==0.19.3 scikit-video==1.1.11 zarr==2.12.0 numcodecs==0.10.2 pygame==2.1.2 pymunk==6.2.1 gym==0.26.2 shapely==1.8.4 huggingface_hub==0.15.1 numpy==1.23.5 opencv-python==4.6.0.66 ipython==8.10.0 gdown==4.7.1"

    ssh -i "$private_ssh_key" "$remote_ssh_user@$remote_ssh_host" "echo 'hugging_face_api_key=$HUGGINGFACE_API_KEY' >> '/home/$remote_ssh_user/.env'"
else
    echo "Skipping requirements installation."
fi

# Run the infernece script on the remote instance
read -p "Would you like to run the inference script on the remote instance? (y/n): " run_inference
if [[ $run_inference == "y" ]]; then

    echo "Running inference script on remote instance..."
    ssh -i "$private_ssh_key" "$remote_ssh_user@$remote_ssh_host" "source .venv/bin/activate && nohup python3 ~/$MAIN_SCRIPT_PATH > push_t_state_policy_experiment_results.log 2>&1 &" &
else
    echo "Skipping inference script execution."
fi