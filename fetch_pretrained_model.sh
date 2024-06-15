# PIXIE pretrained model and utilities
echo -e "\nYou need to register at https://pixie.is.tue.mpg.de/"
read -p "Username (PIXIE):" username
read -p "Password (PIXIE):" password
username=$(urle $username)
password=$(urle $password)
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=pixie&sfile=pixie_model.tar&resume=1' -O './data/pixie_model.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=pixie&sfile=utilities.zip&resume=1' -O './data/utilities.zip' --no-check-certificate --continue


echo -e "\nDownloading more data..."
wget https://nextcloud.tuebingen.mpg.de/index.php/s/jraekdRrxCzYEWB/download -O ./data/delta_utilities2.zip
unzip ./data/delta_utilities2.zip -d ./data
mv ./data/delta_utilities2/* ./data/
rm ./data/delta_utilities2.zip
rm -rf ./data/delta_utilities2

# trained model for RobustVideoMatting
mkdir -p assets/MODNet

# if failed, please download the model from
# https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR
echo -e "Downloading MODNet model..."
FILEID=1Nf1ZxeJZJL8Qx9KadcYYyEmmlKhTADxX
FILENAME=assets/MODNet/modnet_webcam_portrait_matting.ckpt.pth
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt

FILEID=1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz
FILENAME=assets/MODNet/modnet_photographic_portrait_matting.ckpt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt


# trained model for face-parsing
# if failed, please download the model from
# https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view
mkdir -p assets/face_parsing
echo -e "Downloading face_parsing model..."
FILEID=154JgKpzCPW82qINcVieuPH3fZ2e0P812
FILENAME=assets/face_parsing/model.pth
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt



