echo "Downloading Vicon room 1..."
curl "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip" -o "v1.zip"
unzip -qq v1.zip -d v1
rm -rf v1.zip
echo "Downloading Vicon room 2..."
curl "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_02_medium/V1_02_medium.zip" -o "v2.zip"
unzip -qq v2.zip -d v2
rm -rf v2.zip
echo "Downloading Vicon room 3..."
curl "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_03_difficult/V1_03_difficult.zip" -o "v3.zip"
unzip -qq v3.zip -d v3
rm -rf v3.zip
echo "Downloading Vicon room 4..."
curl "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_01_easy/V2_01_easy.zip" -o "v4.zip"
unzip -qq v4.zip -d v4
rm -rf v4.zip
echo "Downloading Vicon room 5..."
curl "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_02_medium/V2_02_medium.zip" -o "v5.zip"
unzip -qq v5.zip -d v5
rm -rf v5.zip
echo "Downloading Vicon room 6..."
curl "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_03_difficult/V2_03_difficult.zip" -o "v6.zip"
unzip -qq v6.zip -d v6
rm -rf v6.zip
