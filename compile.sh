#!/bin/bash

$HADOOP_HOME/sbin/start-all.sh

$classpath = #add classpath here 

mkdir KMeansJar

javac -classpath $classpath -d ./KMeansJar KMeans.java

jar -cvf kmeans.jar -C ./KMeansJar .

rm -r KMeansJar

