# KMeans Clustering using MapReduce Framework

## Steps to Start System in Local Machine :computer:

1. Make sure to install ```openjdk```, ```openssh-server```, and ```openssh-client```

2. Download the latest version of hadoop from Apache's website.

3. Follow the instruction to setup hadoop on a single node

4. Use the command ```hadoop classpath``` to get the class path and add its a variable in the ```complie.sh``` file 

### System used for Development and Testing 
- i7-1170GH
- 8 GB RAM
- OpenJDK 8
- Hadoop 3.2.4

## For deployed hadoop enviroments :globe_with_meridians: 

When using a networked deployed hadoop enviroment, use the command ```hadoop classpath``` of the deployement to compile the 


## Running mapreduce 
1. Run the following file to compile the Mapreduce Program

```./complie.sh```

If unable to execute, give necessary executable permission. eg. ``` chmod +x compile.sh ```

2. Move the dataset use ```put``` to HDFS

3. Run the following command to execute the program

```hadoop jar kmeans.jar /path/to/dataset /path/to/output number_of_clusters```

## Possible Improvements
1. Needs testing at scale, as the Overhead was too much in a smaller system with limited processing power, as the number of node the system used was also limited

2. Need help in finding out how to use the a deployed hadoop enviroment, as the instructions are very vague


## Tasks :white_check_mark:
 - [X] Create the Mapper Program

 - [X] Create the Reducer Program

 - [X] Using a Reducer to reduce network overheadL

