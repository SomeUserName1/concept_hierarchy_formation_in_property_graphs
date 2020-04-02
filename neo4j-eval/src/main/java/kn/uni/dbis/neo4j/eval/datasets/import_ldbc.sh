git #!/bin/bash

which docker
if [ $? == 1 ]; then
  # Setup docker
  sudo apt-get remove docker docker-engine docker.io containerd runc
  sudo apt-get update
  sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
  sudo add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
	   $(lsb_release -cs) \
	   stable"
  sudo apt-get update
  sudo apt-get install docker-ce docker-ce-cli containerd.io

  sudo groupadd docker
  sudo usermod -aG docker $USER
  newgrp docker
fi

which neo4j
if [ $? == 1 ]; then
  # install neo4j
  sudo add-apt-repository -y ppa:openjdk-r/ppa
  sudo apt-get update
  sudo apt-get install openjdk-8-jdk openjdk-8-demo openjdk-8-doc openjdk-8-jre-headless openjdk-8-source
  sudo update-java-alternatives --jre --set java-1.8.0-openjdk-amd64
  wget -O - https://debian.neo4j.org/neotechnology.gpg.key | sudo apt-key add -
  echo 'deb https://debian.neo4j.org/repo stable/' | sudo tee -a /etc/apt/sources.list.d/neo4j.list
  sudo apt-get update
  sudo apt-get install neo4j=1:3.5.9
  echo "dbms.security.auth_enabled=false" >>/etc/neo4j/neo4j.conf
fi

# Generate the ldbc snb data set
if [ ! -d "ldbc_snb_datagen" ]; then
  git clone git@github.com:ldbc/ldbc_snb_datagen.git
fi
cd ldbc_snb_datagen

if [ ! -d "out/social_network" ]; then
  echo "ldbc.snb.datagen.generator.scaleFactor:snb.interactive.1
    ldbc.snb.datagen.serializer.personSerializer:ldbc.snb.datagen.serializer.snb.interactive.CSVCompositePersonSerializer
    ldbc.snb.datagen.serializer.invariantSerializer:ldbc.snb.datagen.serializer.snb.interactive.CSVCompositeInvariantSerializer
    ldbc.snb.datagen.serializer.personActivitySerializer:ldbc.snb.datagen.serializer.snb.interactive.CSVCompositePersonActivitySerializer" >params.ini

  docker build . --tag ldbc/datagen
  mkdir out
  docker run --rm --mount type=bind,source="$(pwd)/out",target="/opt/ldbc_snb_datagen/out" --mount type=bind,source="$(pwd)/params.ini",target="/opt/ldbc_snb_datagen/params.ini" ldbc/datagen
fi
cd out
sudo chown -R $USER:$USER social_network/ substitution_parameters/

# convert dataset and import
export NEO4J_DATA_DIR=$(pwd)/social_network
export NEO4J_DB_DIR=/var/lib/neo4j/data/databases/ldbc_snb.db
export POSTFIX=_0_0.csv

echo "starting preprocessing"

# replace headers
while read line; do
  IFS=' ' read -r -a array <<<$line
  filename=${array[0]}
  header=${array[1]}
  sed -i "1s/.*/$header/" "${NEO4J_DATA_DIR}/${filename}${POSTFIX}"
done < <(curl "https://raw.githubusercontent.com/ldbc/ldbc_snb_implementations/master/cypher/load-scripts/headers.txt")

# replace labels with one starting with an uppercase letter
sed -i "s/|city$/|City/" "${NEO4J_DATA_DIR}/place${POSTFIX}"
sed -i "s/|country$/|Country/" "${NEO4J_DATA_DIR}/place${POSTFIX}"
sed -i "s/|continent$/|Continent/" "${NEO4J_DATA_DIR}/place${POSTFIX}"
sed -i "s/|company|/|Company|/" "${NEO4J_DATA_DIR}/organisation${POSTFIX}"
sed -i "s/|university|/|University|/" "${NEO4J_DATA_DIR}/organisation${POSTFIX}"

# convert each date of format yyyy-mm-dd to a number of format yyyymmddd
sed -i "s#|\([0-9][0-9][0-9][0-9]\)-\([0-9][0-9]\)-\([0-9][0-9]\)|#|\1\2\3|#g" "${NEO4J_DATA_DIR}/person${POSTFIX}"

# convert each datetime of format yyyy-mm-ddThh:mm:ss.mmm+0000
# to a number of format yyyymmddhhmmssmmm
sed -i "s#|\([0-9][0-9][0-9][0-9]\)-\([0-9][0-9]\)-\([0-9][0-9]\)T\([0-9][0-9]\):\([0-9][0-9]\):\([0-9][0-9]\)\.\([0-9][0-9][0-9]\)+0000#|\1\2\3\4\5\6\7#g" ${NEO4J_DATA_DIR}/*${POSTFIX}

echo "preprocessing finished"

sudo service neo4j stop
sudo rm -rf $NEO4J_DB_DIR

sudo neo4j-admin import --database=$(basename $NEO4J_DB_DIR) \
  --id-type=INTEGER \
  --nodes:Message:Comment "${NEO4J_DATA_DIR}/comment${POSTFIX}" \
  --nodes:Forum "${NEO4J_DATA_DIR}/forum${POSTFIX}" \
  --nodes:Organisation "${NEO4J_DATA_DIR}/organisation${POSTFIX}" \
  --nodes:Person "${NEO4J_DATA_DIR}/person${POSTFIX}" \
  --nodes:Place "${NEO4J_DATA_DIR}/place${POSTFIX}" \
  --nodes:Message:Post "${NEO4J_DATA_DIR}/post${POSTFIX}" \
  --nodes:TagClass "${NEO4J_DATA_DIR}/tagclass${POSTFIX}" \
  --nodes:Tag "${NEO4J_DATA_DIR}/tag${POSTFIX}" \
  --relationships:HAS_CREATOR "${NEO4J_DATA_DIR}/comment_hasCreator_person${POSTFIX}" \
  --relationships:IS_LOCATED_IN "${NEO4J_DATA_DIR}/comment_isLocatedIn_place${POSTFIX}" \
  --relationships:REPLY_OF "${NEO4J_DATA_DIR}/comment_replyOf_comment${POSTFIX}" \
  --relationships:REPLY_OF "${NEO4J_DATA_DIR}/comment_replyOf_post${POSTFIX}" \
  --relationships:CONTAINER_OF "${NEO4J_DATA_DIR}/forum_containerOf_post${POSTFIX}" \
  --relationships:HAS_MEMBER "${NEO4J_DATA_DIR}/forum_hasMember_person${POSTFIX}" \
  --relationships:HAS_MODERATOR "${NEO4J_DATA_DIR}/forum_hasModerator_person${POSTFIX}" \
  --relationships:HAS_TAG "${NEO4J_DATA_DIR}/forum_hasTag_tag${POSTFIX}" \
  --relationships:HAS_INTEREST "${NEO4J_DATA_DIR}/person_hasInterest_tag${POSTFIX}" \
  --relationships:IS_LOCATED_IN "${NEO4J_DATA_DIR}/person_isLocatedIn_place${POSTFIX}" \
  --relationships:KNOWS "${NEO4J_DATA_DIR}/person_knows_person${POSTFIX}" \
  --relationships:LIKES "${NEO4J_DATA_DIR}/person_likes_comment${POSTFIX}" \
  --relationships:LIKES "${NEO4J_DATA_DIR}/person_likes_post${POSTFIX}" \
  --relationships:IS_PART_OF "${NEO4J_DATA_DIR}/place_isPartOf_place${POSTFIX}" \
  --relationships:HAS_CREATOR "${NEO4J_DATA_DIR}/post_hasCreator_person${POSTFIX}" \
  --relationships:HAS_TAG "${NEO4J_DATA_DIR}/comment_hasTag_tag${POSTFIX}" \
  --relationships:HAS_TAG "${NEO4J_DATA_DIR}/post_hasTag_tag${POSTFIX}" \
  --relationships:IS_LOCATED_IN "${NEO4J_DATA_DIR}/post_isLocatedIn_place${POSTFIX}" \
  --relationships:IS_SUBCLASS_OF "${NEO4J_DATA_DIR}/tagclass_isSubclassOf_tagclass${POSTFIX}" \
  --relationships:HAS_TYPE "${NEO4J_DATA_DIR}/tag_hasType_tagclass${POSTFIX}" \
  --relationships:STUDY_AT "${NEO4J_DATA_DIR}/person_studyAt_organisation${POSTFIX}" \
  --relationships:WORK_AT "${NEO4J_DATA_DIR}/person_workAt_organisation${POSTFIX}" \
  --relationships:IS_LOCATED_IN "${NEO4J_DATA_DIR}/organisation_isLocatedIn_place${POSTFIX}" \
  --delimiter '|'

sudo chown -R neo4j:neo4j $NEO4J_DB_DIR

sudo service neo4j restart

cd ../..
rm -r ldbc_snb_datagen
