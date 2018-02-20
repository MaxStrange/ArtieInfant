# This script should be run on the master node everytime (after) you run nodebringup.sh
set -e

if [ "$#" -ne 2 ]; then
    echo "USAGE: $0 <NEW_NODE_IP> <NEW_NODE_HOSTNAME>"
    exit 1
fi

# Add new node to slaves file
echo $2 >> /opt/hadoop/etc/hadoop/slaves

# Add new hode to hosts file
sudo echo "$1   $2" >> /etc/hosts

echo "Now do the following things manually:"
echo "1) cat .ssh/id_rsa.pub | ssh max@$2 'cat >> .ssh/authorized_keys' from EACH node"
echo "2) rm -rf /hdfs/tmp/* on EACH node"
echo "3) hdfs namenode -format (and make sure to answer any questions with a CAPITAL Y)"
echo ""
echo "Once all of that is done, come back to the main node and run:"
echo "start-hdfs.sh"
echo "start-yarn.sh"
echo ""
echo "Then run 'hdfs dfsadmin -report' and check to make sure it can see the correct number of nodes."
