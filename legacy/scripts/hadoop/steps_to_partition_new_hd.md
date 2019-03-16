## Requisitioning a New Mini PC into the Hadoop Cluster

These are the steps to get a new node up and running as part of the Hadoop cluster.

### Setting up the hardware

1. Hardware: AcePC AK1; seagate 1TB Barricuda internal hard drive
1. Unscrew the bottom plate on the AcePC and insert the hard drive. It should click, more or less. Put the cover back on.
1. Plug the node into the ethernet switch, powersupply, screen, keyboard, and mouse.
1. Plug an Ubuntu 16.04 USB stick in.
1. Power on the device and hit Esc a few times as the AcePC logo pops up to enter the UEFI.
1. Change the boot order to boot into the USB stick first. Note that it is possible that you will end up in Windows at some point - you may even have to the first time. You'll have to just sit through the installation in that case, and enter the UEFI from Windows possibly.
1. If the USB stick is not recognized, you will need to use a respun image of Ubuntu 16.04 - use the linuxium isorespin.sh script that you can find on the internet. Follow Linuxium's tutorial on how to respin for Apollo Lake processors.
1. Write a USB with the new iso and plug that into the AcePC and boot through the UEFI.
1. This time Ubuntu should be recognized on the USB stick.
1. Install as normal, but DO NOT use Logical Volume Management.

### Setting up Ubuntu

1. Put this repo's bringup.sh script (should be one directory up if things haven't changed) on to a USB stick and copy it into the AcePC.
1. Make it executable and run it.
1. Follow any instructions it prompts at the end.
1. Reserve this node's IP address in the router and add its hostname to any /etc/hosts files you want in the rest of the nodes in the system. Do not do this for the hadoop cluster nodes yet.

### Setting up Hadoop

1. Run this directory's nodebringup.sh script.
1. Follow any prompts it has at the end of the script.
1. In particular, you will need to partition the fairly newly installed internal harddrive and mount it at startup under /hdfs/tmp. To do this, follow these steps:
  1. Determine drive info by running lsblk and seeing what its called (sda probably)
  1. Run 'sudo fdisk /dev/sda'
  1. n for new partition
  1. p for primary partition
  1. 1 for the only drive number
  1. enter for any defaults
  1. w to write to the disk
  1. Now the disk should be partitioned. Let's format it: 'sudo mkfs -t ext4 /dev/sda1'
  1. Check the device's UUID with 'sudo blkid'
  1. Add the following to /etc/fstab: "/dev/sda1    /hdfs/tmp    ext4    defaults    0    2"

Reboot and you should now have another node in the Hadoop cluster.
