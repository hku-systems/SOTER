# Artifact for paper #1520 SOTER: Guarding Black-box Inference for General Neural Networks at the Edge

## Artifact summary

SOTER is a TEE-based secure inference system that protects model
providers' private models to deploy and run on top of untrusted third-party
devices. SOTER embraces several key features: model confidentiality,
integrity, low inference latency and high accuracy simultaneously. This artifact contains the implementation of SOTER and scripts for reproducing the main results of this work.

## Artifact Check-list

- Code link: <https://github.com/hku-systems/SOTER>
- OS Version: Ubuntu 18.04 or 20.04.
- Linux kernel version: >= 5.10
- Python version: >= 3.8.5.
- TEE code base: [Graphene SGX](https://github.com/gramineproject/graphene/blob/master/Documentation/building.rst)
- Metrics: latency and accuracy.
- Expected runtime: see the documents or runtime logs for each experiment.

## Experiments

If otherwise specified, all SOTER's experiments run on top of an SGX virtual
machine and a GPU machine. Each experiment runs 1000 inferences by default.

### Prepare

1. Please login to our cluster following [this page](https://github.com/hku-systems/SOTER/blob/main/servers.md).

2. Please open two terminals (`T1` and `T2`). Use
   `T1` to login to the SGX VM (ubuntu@192.168.122.158) and go to
   `/home/ubuntu/atc22-artifact/SOTER`; use `T2` to login to the GPU machine
   (xian@202.45.128.185) and go to `/home/xian/atc22-artifact/SOTER`. We have
   set up all necessary environments to run all the experiments.

3. Each experiment will generate a figure in the `./figure` directory of this
   repository. You can fetch generated figures to your email by
   running `echo "Figures" | mail -s "Experiments" $YOUR_EMAIL -A $FIGURE` in
   the `./figure` directory，e.g., `echo "Figure8" | mail -s
   "Fingerprint experiments" xxx@gmail.com -A figure8a-oblifp.pdf`. Alternatively, you can
   also use `scp` to send figures to your computer.

Please be noted that different evaluators cannot run experiments at the same
time. Running experiments at the same
time may downgrade the performance due to the shared SGX/GPU resources. You may check whether other evaluators are running the
experiments before you run the artifact.

### Kick-off Functional

#### SOTER-VGG: A Kick-off Functional Experiment

**Step 1:**
Login to the SGX VM and GPU machine with two terminals in our cluster. 

**Step 2:**
In the GPU terminal, run the following commands to compile and start the
GPU server:

```shell
cd /home/xian/atc22-artifact/SOTER/vgg-partition/cpp/soter-graphene-vgg

bash runserver.sh
```

**Step 3:**
In the SGX VM terminal, run the following commands to compile and start the
client in SGX hardware mode:

```shell
cd /home/ubuntu/atc22-artifact/SOTER/graphene-vgg-partition

bash runclient.sh
```

**Output:**

```shell
For 1000 inferences ...
Time elapsed: 42286 ms.
Time consuming: 43.286 ms per instance.
Completed successfully !!!
```

If you see the the above logs, the artifact runs perfectly.
 


### Experiment 1: End-to-end performance

#### Experiment 1-1: Performance of 

**Command to run:**

```shell
bash run.sh performance
```

**Output:**

- 

**Expected results:**

- 
