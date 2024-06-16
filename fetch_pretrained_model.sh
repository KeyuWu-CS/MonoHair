# trained model for RobustVideoMatting
mkdir -p assets/MODNet
echo -e "Downloading MODNet model..."
# https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR
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


