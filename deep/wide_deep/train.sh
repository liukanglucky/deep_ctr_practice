#!/bin/bash

export HADOOP_USER_NAME=
export HADOOP_USER_PASSWORD=

source $HADOOP_HOME/libexec/hadoop-config.sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$JAVA_HOME/jre/lib/amd64/server
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

CLASSPATH=$($HADOOP_HDFS_HOME/bin/hdfs classpath --glob) python3 -u wdl_train.py