#!/usr/bin/bash

## downloads and pre-processes GL18 dataset for training

# 1. Download sub tarballs
wget -nc https://imperialcollegelondon.box.com/shared/static/lbk9qwtn438a0edm5g52fv0az4o8s6ml.partaa -O data/train/gl18/GL18_TRAIN.partaa
wget -nc https://imperialcollegelondon.box.com/shared/static/arnssx7f80vq6lhsufxrwl7xhxixpasj.partab -O data/train/gl18/GL18_TRAIN.partab
wget -nc https://imperialcollegelondon.box.com/shared/static/p2zlm5gobfsk85qu2evyv0q2wnu4vr4v.partac -O data/train/gl18/GL18_TRAIN.partac
wget -nc https://imperialcollegelondon.box.com/shared/static/38uo0hgxl7dmnxizzfm0t828sba1dznv.partad -O data/train/gl18/GL18_TRAIN.partad
wget -nc https://imperialcollegelondon.box.com/shared/static/s1ig9xfs95vvix1zvf2t9f76cydlhntq.partae -O data/train/gl18/GL18_TRAIN.partae
wget -nc https://imperialcollegelondon.box.com/shared/static/0x3idlrbbkl3j15g16tm9tt8qjybl2r2.partaf -O data/train/gl18/GL18_TRAIN.partaf
wget -nc https://imperialcollegelondon.box.com/shared/static/1barrmdjeh9552zhqh1zm2kvoike9a7f.partag -O data/train/gl18/GL18_TRAIN.partag
wget -nc https://imperialcollegelondon.box.com/shared/static/dt9v439bmb2ckl12zulgjznqaeboujdb.partah -O data/train/gl18/GL18_TRAIN.partah
wget -nc https://imperialcollegelondon.box.com/shared/static/358t3dcewmosuw7qz4oal59xjifzzq40.partai -O data/train/gl18/GL18_TRAIN.partai
wget -nc https://imperialcollegelondon.box.com/shared/static/jgqm8mxjjcjx68fpjgdafzdybdt56m52.partaj -O data/train/gl18/GL18_TRAIN.partaj
wget -nc https://imperialcollegelondon.box.com/shared/static/0uatr2u794xz513o2uofrpl24rh7gml8.partak -O data/train/gl18/GL18_TRAIN.partak
wget -nc https://imperialcollegelondon.box.com/shared/static/0j7bfq8f8ijalptl0dwt0nsozbslfmqf.partal -O data/train/gl18/GL18_TRAIN.partal
wget -nc https://imperialcollegelondon.box.com/shared/static/mt04r82bl42du5sl03qkn0ms27w59qn4.partam -O data/train/gl18/GL18_TRAIN.partam
wget -nc https://imperialcollegelondon.box.com/shared/static/dyg8txwd2recydersmpyu8y7n6fqszi3.partan -O data/train/gl18/GL18_TRAIN.partan
wget -nc https://imperialcollegelondon.box.com/shared/static/mopb621p8lnud9m1foq98xkuqwg3a6z5.partao -O data/train/gl18/GL18_TRAIN.partao
wget -nc https://imperialcollegelondon.box.com/shared/static/tpyuaj26un74nv5xl3qhljrjyph73af5.partap -O data/train/gl18/GL18_TRAIN.partap
wget -nc https://imperialcollegelondon.box.com/shared/static/8qjbotgdzj1jrt7i79yv9ubag91wn6aj.partaq -O data/train/gl18/GL18_TRAIN.partaq
wget -nc https://imperialcollegelondon.box.com/shared/static/et02ppjjm41eebutpnv2gzdaz7iadatq.partar -O data/train/gl18/GL18_TRAIN.partar
wget -nc https://imperialcollegelondon.box.com/shared/static/zu44mz608syhb9uy9thnkibeof72dg3p.partas -O data/train/gl18/GL18_TRAIN.partas
wget -nc https://imperialcollegelondon.box.com/shared/static/myq5vlpyqu927ffpikn1vqw059oy38k8.partat -O data/train/gl18/GL18_TRAIN.partat
wget -nc https://imperialcollegelondon.box.com/shared/static/z7cz70tj8akf2pn8otbtcggq6k75oktr.partau -O data/train/gl18/GL18_TRAIN.partau
wget -nc https://imperialcollegelondon.box.com/shared/static/qwsairfpb25fg4vtmecykfq5qahb9j3v.partav -O data/train/gl18/GL18_TRAIN.partav

# 2. cat them back into single file
cd data/train/gl18/
cat GL18_TRAIN.partaa GL18_TRAIN.partab GL18_TRAIN.partac GL18_TRAIN.partad GL18_TRAIN.partae GL18_TRAIN.partaf GL18_TRAIN.partag GL18_TRAIN.partah GL18_TRAIN.partai GL18_TRAIN.partaj  GL18_TRAIN.partak GL18_TRAIN.partal GL18_TRAIN.partam GL18_TRAIN.partan GL18_TRAIN.partao GL18_TRAIN.partap GL18_TRAIN.partaq GL18_TRAIN.partar GL18_TRAIN.partas GL18_TRAIN.partat GL18_TRAIN.partau GL18_TRAIN.partav > GL18_TRAIN.tar.gz 

# 3. remove sub-tarballs and extract
rm -rf GL18_TRAIN.parta*
tar -xf GL18_TRAIN.tar.gz
rm GL18_TRAIN.tar.gz

# 4. rename folder containing images
mv train/ jpg/

# 5. download train/val labels
wget -nc https://imperialcollegelondon.box.com/shared/static/izj2yt4nizytt542is42wae87fg4k6sv.csv -O train.csv
wget -nc https://imperialcollegelondon.box.com/shared/static/2b3vhx3unjldu27p1m31z4vwfnkvcops.csv -O boxes_split1.csv
wget -nc https://imperialcollegelondon.box.com/shared/static/d2ui9mjwcvwa6f17gaoca6w6sr7tjmzg.csv -O boxes_split2.csv

# 6. create the db pickle file
python3 create_db_pickle.py