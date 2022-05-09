# Getting started: access our servers for artifact evaluation

## Steps

### Step 1: Connect to HKU CS gatekeeper

For security reasons, our department only allow access to the gatekeeper using SSH keys.

1. Create a new file in your computer, name it as `soterkey`. Copy the private
   key from our HotCRP response into `soterkey`.

2. `chmod 400 ./soterkey`.

3. `ssh -i ./soterkey atc22@gatekeeper3.cs.hku.hk`.

### Step 2: Connect to our cluster

#### How to connect to the GPU server

    ssh [server_account_1]@202.45.128.185

#### How to connect to the SGX virtual machine (using 3 commands below)

    ssh [server_account_1]@202.45.128.185

    ssh [server_account_2]@10.22.1.16

    ssh ubuntu@192.168.122.158

`server_account_1`, `server_account_2` and their corresponding passwords are
listed in our HotCRP response. Step 3 does not need a password.


Note that, since the environment is not completely clean, the experimental results may have some outliers if other people happen to run tasks on the cluster at the same time.
**You may re-run the outlier data points or simply ignore them when such contention happens**.

## After accessing the cluster

We have helped to finish the setup for meeting prerequisite requirements and
creating the working environment for GPU computation and Graphene-SGX, so you can skip these steps in the instructions.

After cloning the codebase, you can skip all Deployment steps. We have marked
the starting point for running this artifact as `Kick-off Functional`.

