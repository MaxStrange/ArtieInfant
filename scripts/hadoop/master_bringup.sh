# This script should be run on the namenode
set -e
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java8-installer

sudo apt-get install    curl \
                        telnet \
                        dnsutils \
                        sed \
                        ack-grep

mkdir -p ~/.ssh
ssh-keygen -t rsa -P ""
cat ~/.ssh/id_rsa.pub > ~/.ssh/authorized_keys

wget http://apache.mirrors.spacedump.net/hadoop/core/stable/hadoop-2.9.0.tar.gz
sudo mkdir -p /opt
sudo tar -xvzf hadoop-2.9.0.tar.gz -C /opt/
sudo mv /opt/hadoop-2.9.0 /opt/hadoop

JAVA_HOME_PATH=$(readlink -f /usr/bin/java | sed "s:bin/java::")
echo "" >> ~/.bashrc
echo "export JAVA_HOME=$JAVA_HOME_PATH" >> ~/.bashrc
echo 'export HADOOP_INSTALL=/opt/hadoop' >> ~/.bashrc
echo 'export PATH=$PATH:$HADOOP_INSTALL/bin' >> ~/.bashrc
echo 'export PATH=$PATH:$HADOOP_INSTALL/sbin' >> ~/.bashrc
echo 'export HADOOP_MAPRED_HOME=$HADOOP_INSTALL' >> ~/.bashrc
echo 'export HADOOP_COMMON_HOME=$HADOOP_INSTALL' >> ~/.bashrc
echo 'export HADOOP_HDFS_HOME=$HADOOP_INSTALL' >> ~/.bashrc
echo 'export YARN_HOME=$HADOOP_INSTALL' >> ~/.bashrc
echo 'export HADOOP_HOME=$HADOOP_INSTALL' >> ~/.bashrc

echo "Sourcing .bashrc"
. ~/.bashrc

sed -i /opt/hadoop/etc/hadoop/hadoop-env.sh -e "s@^export JAVA_HOME=.*@export JAVA_HOME=$JAVA_HOME_PATH@"

echo "
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://$HOSTNAME:54310</value>
    </property>
    <property>
        <name>hadoop.tmp.dir</name>
        <value>/hdfs/tmp</value>
    </property>
</configuration>
" > /opt/hadoop/etc/hadoop/core-site.xml

echo '
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>2</value>
    </property>
    <property>
        <name>dfs.blocksize</name>
        <value>5242880</value>
    </property>
</configuration>
' > /opt/hadoop/etc/hadoop/hdfs-site.xml

echo '
<configuration>
  <property>
    <name>mapreduce.framework.name</name>
    <value>yarn</value>
  </property>
  <property>
    <name>mapreduce.map.memory.mb</name>
    <value>256</value>
  </property>
  <property>
    <name>mapreduce.map.java.opts</name>
    <value>-Xmx210m</value>
  </property>
  <property>
    <name>mapreduce.reduce.memory.mb</name>
    <value>256</value>
  </property>
  <property>
    <name>mapreduce.reduce.java.opts</name>
    <value>-Xmx210m</value>
  </property>
  <property>
    <name>yarn.app.mapreduce.am.resource.mb</name>
    <value>256</value>
  </property>
</configuration>
' > /opt/hadoop/etc/hadoop/mapred-site.xml

echo "
<configuration>
<!-- Site specific YARN configuration properties -->
  <property>
    <name>yarn.nodemanager.aux-services</name>
    <value>mapreduce_shuffle</value>
  </property>
  <property>
    <name>yarn.nodemanager.resource.cpu-vcores</name>
    <value>4</value>
  </property>
  <property>
    <name>yarn.nodemanager.resource.memory-mb</name>
    <value>4096</value>
  </property>
  <property>
    <name>yarn.scheduler.minimum-allocation-mb</name>
    <value>128</value>
  </property>
  <property>
    <name>yarn.scheduler.maximum-allocation-mb</name>
    <value>4096</value>
  </property>
  <property>
    <name>yarn.scheduler.minimum-allocation-vcores</name>
    <value>1</value>
  </property>
  <property>
    <name>yarn.scheduler.maximum-allocation-vcores</name>
    <value>4</value>
  </property>
  <property>
    <name>yarn.nodemanager.vmem-check-enabled</name>
    <value>false</value>
    <description>Whether virtual memory limits will be enforced for containers</description>
  </property>
  <property>
    <name>yarn.nodemanager.vmem-pmem-ratio</name>
    <value>4</value>
    <description>Ratio between virtual memory to physical memory when setting memory limits for containers</description>
  </property>
  <property>
    <name>yarn.resourcemanager.resource-tracker.address</name>
    <value>$HOSTNAME:8025</value>
  </property>
  <property>
    <name>yarn.resourcemanager.scheduler.address</name>
    <value>$HOSTNAME:8030</value>
  </property>
  <property>
    <name>yarn.resourcemanager.address</name>
    <value>$HOSTNAME:8040</value>
  </property>
</configuration>
" > /opt/hadoop/etc/hadoop/yarn-site.xml

# Create the HDFS file system
sudo mkdir -p /hdfs/tmp
sudo chmod 750 /hdfs/tmp
hdfs namenode -format
