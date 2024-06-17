echo -e "\nDownloading more data..."
wget https://nextcloud.tuebingen.mpg.de/index.php/s/jraekdRrxCzYEWB/download -O ./assets/data/delta_utilities2.zip
unzip ./assets/data/delta_utilities2.zip -d ./assets/data
mv ./assets/data/delta_utiliites2/* ./assets/data/
rm ./assets/data/delta_utilities2.zip
rm -rf ./assets/data/delta_utilities2

# detla utilities
echo -e "\nDownloading delta data..."
wget https://nextcloud.tuebingen.mpg.de/index.php/s/zR3DM3zEdje984c/download -O ./assets/data/delta_utilities.zip
unzip ./assets/data/delta_utilities.zip -d ./assets/data
mv ./assets/data/delta_utilities/* ./assets/data/
rm ./assets/data/delta_utilities.zip
rm -rf ./assets/data/delta_utilities
rm -rf ./assets/data/__MACOSX

# SMPL-X 2020 (neutral SMPL-X model with the FLAME 2020 expression blendshapes)
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }
echo -e "\nYou need to register at https://smpl-x.is.tue.mpg.de"
read -p "Username (SMPL-X):" username
read -p "Password (SMPL-X):" password
username=$(urle $username)
password=$(urle $password)
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=SMPLX_NEUTRAL_2020.npz&resume=1' -O './assets/data/SMPLX_NEUTRAL_2020.npz' --no-check-certificate --continue

# PIXIE pretrained model and utilities
echo -e "\nYou need to register at https://pixie.is.tue.mpg.de/"
read -p "Username (PIXIE):" username
read -p "Password (PIXIE):" password
username=$(urle $username)
password=$(urle $password)
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=pixie&sfile=pixie_model.tar&resume=1' -O './assets/data/pixie_model.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=pixie&sfile=utilities.zip&resume=1' -O './assets/data/utilities.zip' --no-check-certificate --continue
cd ./assets/data
unzip utilities.zip
rm utilities.zip
cd ../..









