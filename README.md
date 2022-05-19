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

If otherwise specified, all SOTER's experiments run on top of a GrapheneSGX-equipped TEE virtual
machine and a 2080 Ti GPU machine. Each experiment runs 1000 inferences by default.

### Prepare

1. Please login to our cluster following [this page](https://github.com/hku-systems/SOTER/blob/main/server.md).

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
cd /home/xian/atc22-artifact/SOTER/mtr-partition/vgg-partition/cpp/soter-graphene-vgg

bash runserver_func.sh
```

**Step 3:**
In the SGX VM terminal, run the following commands to compile and start the
SGX client:

```shell
cd /home/ubuntu/atc22-artifact/SOTER/teertconfig/graphene-vgg-partition

bash runclient_func.sh
```

**Output in SGX VM:**

```shell
For 1000 inferences ...
Time elapsed: 42286 ms.
Time consuming: 42.286 ms per instance.
Completed successfully !!!
```

If you see the the above logs (the "**Time consuming**" is normal to fluctuate
within -5 ~ +5 ms on VGG19 owing to the nature of inference tasks), the artifact runs
perfectly. 

### Experiment 1: End-to-end performance (45 mins)

This experiment runs SOTER and three secure inference baselines (i.e., AegisDNN,
eNNclave, and MLCapsule) on six neural networks. The output reports each
system's inference latency normalized to insecure GPU inference. 

**Command to run (in the GPU terminal):**

```shell
bash ~/atc22-artifact/SOTER/script/latency_fig5.sh
```

**Output:**

- A pdf file named `normalized_latency.pdf` in `~/atc22-artifact/SOTER/figure/`, containing the normalized inference
 latency of SOTER and all baselines.
- You can also find the log files for generating figure in
  `~/atc22-artifact/SOTER/script/data/`

**Expected results:**

- MLCapsule incurs the highest latency (i.e., highest bar) among all systems and models.
- For VGG19: partition-based systems (i.e., SOTER, AegisDNN, and eNNclave)
  have similar inference latency, which should be lower than MLCapsule.
- For other five models: SOTER incurs slightly higher latency than AegisDNN;
  eNNclave is not applicable to these five models (i.e., N/A in figure).

**Important notes for Experiment 1:**

- SOTER's inference latency on six models (i.e., numbers on each red bar) may not
  exactly match the results presented in the paper due to the instability nature
  of model inference. 
- This experiment is expected to take around 45 mins before producing the final
  PDF file, as we need to compile and run each program. Please do not interrupt
  the running process. If you failed in this step, please make sure
  other evaluators are not running this experiment concurrently.



### Experiment X: The pattern between SOTER's fingerprint protocol and fixed fingerprint (baseline)

This experiment runs two VGG inference programs (i.e., one for SOTER's
fingerprint protocol, one for a fixed fingerprint protocol as a
baseline), calculates and compares the l2 distance between fingerprints and
reports the pattern from an attacker's perspective.

**Command to run (in the GPU terminal):**

```shell
bash /home/xian/atc22-artifact/SOTER/script/run-fpcheck.sh
```

**Output:**

- Two pdf files named `figure8a-oblifp.pdf` and `figure8b-fixedfp.pdf`,
  containing the l2 distance distribution of SOTER and baseline.
- You can also find the log files for generating figures in
  `/home/xian/atc22-artifact/SOTER/fingerprint_obliviousness/figure8a(or
  8b)/l2_dist_obli_fp.dat (or l2_dist_fixed_fp.dat)`

**Expected results:**

- SOTER's fingerprint l2 distance distribution (i.e., `figure8a-oblifp.pdf`) is close to a
  normal distribution.
- Baseline's fingerprint l2 distance distribution (i.e., `figure8b-fixedfp.pdf`)
  has spikes at a small distance and does not share the form of normal distribution.
