#!/usr/bin/env python
# -*- coding: utf-8 -*-



打造本地docker练手环境
1）	本地装虚拟机（推荐用virtualbox），注意要配置双网卡，一块配NAT网络，一块配桥接网络（要两块网卡是因为公司坑爹的网络管控）。
 
 
2）	给虚拟机装ubunu 14.04（或更高版本）64bit系统（其他linux系统也行，但后面的操作就需要自行baidu调整了）。
3）	配置桥接网络的那块网卡地址为静态，注意不要配置网关
配置命令：
sudo vi /etc/network/interfaces
然后参照下图改，注意紫红色的IP地址改成自己工作网段的空闲地址
 
改好后记得reboot生效
4）	修改apt-get的官方源为hikvision源，方法如下
先用cat /etc/apt/sources.list看看自己默认的apt-get源地址
 
比如上图中源地址都是cn.archive.ubuntu.com（英文版ubuntu有可能是us.archive.ubuntu.com，有的甚至是archive.ubuntu.com）。
确定好源地址后，再用下面的命令更新源（红色部分自行替换成上面查出的源地址）
sudo sed -i 's/cn.archive.ubuntu.com/mirrors.hikvision.com.cn/g' /etc/apt/sources.list
5）	添加证书：
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
6）	添加daocloud第三方apt加速源：
echo "deb [arch=amd64] https://get.daocloud.io/docker/apt-repo ubuntu-trusty main" | sudo tee /etc/apt/sources.list.d/docker.list
7）	安装最新版本docker：
sudo apt-get update && sudo apt-get install -y docker-engine
安装完后注意重新login或reboot
8）	用户加入docker组：
sudo usermod -aG docker $USER
9）	添加aliyun第三方docker加速器：
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": ["https://65lyi7el.mirror.aliyuncs.com"]
}
EOF
完毕后重启docker服务生效
sudo service docker restart

