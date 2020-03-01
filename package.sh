#!/usr/bin/env bash

NAME=$1
TAR=$NAME.tar.gz
FILES=(run clean *.md *.txt *.pdf src tests)

if [[ -z $NAME ]]; then
	echo "usage: package.sh <program name>"
	exit -1
fi

rm -rf $TAR ~/Downloads/$TAR

mkdir -p $NAME

for file in ${FILES[@]}; do
  cp -r $file $NAME
done

uperm -c -y -r
tar -czvf $TAR $NAME
mv $TAR ~/Downloads
uperm -d ~/Downloads -y -r
rm -rf $NAME
