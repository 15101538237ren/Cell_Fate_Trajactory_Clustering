#!/bin/bash

BASE_DIR=/Users/emmanueldollinger/PycharmProjects/Cell_Fate_Trajactory_Clustering

STAGE_DATA=$BASE_DIR/DATA/stage_data
STAGE_DATA_PROB2D=$BASE_DIR/DATA/stage_prob2d_data

cd $STAGE_DATA

for f in *; do
    if [ -d "$f" ]; then
    	dir_name=$f
    	mv $STAGE_DATA_PROB2D/$dir_name/* $STAGE_DATA/$dir_name/
    fi
done