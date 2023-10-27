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

1. Please login to our cluster following [this page]().

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

### Major Claims for Artifact Evaluation
- **(C1)** The TEE-shielding baseline (i.e., MLCapsule) incurs the highest inference
  latency among all systems and all six models (Figure 5 in our paper).
- **(C2)** Partition-based systems (i.e., SOTER, AegisDNN, and eNNclave) have
  similar inference latency on VGG19 (Figure 5 in our paper).
- **(C3)** SOTER incurs slightly higher latency than AegisDNN for other five
  models (Figure 5 in our paper).
- **(C4)** With a larger input shape, SOTER's inference latency increases (Figure 6 in our paper).
- **(C5)** With a larger partition ratio, SOTER's inference latency decreases in
  most cases (Figure 6 in our paper).
- **(C6)** With a larger partition ratio, SOTER achieves similar accuracy (i.e.,
  similar confidentiality guarantee) as eNNclave (Figure 7a in our paper).
- **(C7)** With a larger partition ratio, SOTER achieves lower accuracy (i.e., stronger confidentiality
 guarantee) than AegisDNN (Figure 7a&7b in our paper).
- **(C8)** When running SOTER's oblivious fingerprint protocol, the l2 distance
  distribution between randomly sampled fingerprints is similar to a normal
  distribution (Figure 8a in our paper).
- **(C9)** When running a baseline fixed fingerprint protocol, the distribution
  has spikes on a small distance, and does not share a similar form of normal distribution (Figure 8b in our paper).

### One Step to Run All Experiments!

To further ease the evaluation, we make a script that automatically
runs all four experiments! You just need two steps:

**Step 1: Run all experiments (~3.5 hours)**

```shell
cd ~/atc22-artifact/SOTER/script/
bash ./oneforall.sh
```

**Step 2: Email figures to your computer**

```shell
cd ~/atc22-artifact/SOTER/script/ 
bash ./fetch.sh xxx@gmail.com (e.g., bash ./fetch.sh stx635@connect.hku.hk)
```

After fetching all these figures, you can verify the abovementioned **Major
Claims** by referring to the **Expected results** and **Important notes** below.

Note that, you can also run each experiment by yourself with the steps described
below.

### Experiment 1: End-to-end performance (45 mins)

This experiment runs SOTER and three baseline systems (i.e., AegisDNN,
eNNclave, and MLCapsule) on six neural networks. The output reports each
system's inference latency normalized to insecure GPU inference. 

**Command to run (in the GPU terminal):**

```shell
cd ~/atc22-artifact/SOTER/script/
bash ./latency_fig5.sh
```

**Output:**

- A PDF file named `normalized_latency.pdf` in `~/atc22-artifact/SOTER/figure/`, containing the normalized inference
 latency of SOTER and all baselines.
- You can also find the log files for generating figure in
  `~/atc22-artifact/SOTER/script/data/`

**Expected results:**

- The TEE-shielding system (i.e., MLCapsule) incurs the highest latency (match **C1**).
- Partition-based systems (i.e., SOTER, AegisDNN, and eNNclave)
  have similar inference latency on VGG19, which should be lower than MLCapsule. (match **C2**).
- SOTER could incur slightly higher latency than AegisDNN; eNNclave is not applicable
  to other models except VGG19 (match **C3**).

**Important notes of Experiment 1:**

- SOTER's inference latency on six models (i.e., numbers on each red bar) may not
  exactly match the results presented in the paper due to the instability nature
  of model inference. 

### Experiment 2: Partition sensitivity (75 mins)

This experiment runs SOTER on VGG19, tests SOTER with different input shape and
with different partition ratio, and finally reports SOTER's inference latency.

**Command to run (in the GPU terminal):**

```shell
cd ~/atc22-artifact/SOTER/script/
bash ./sensitivity_fig6.sh
``` 

**Output:**

- Two pdf files named `senvgg.pdf` and `sentrans.pdf` in `~/atc22-artifact/SOTER/figure/`,
  containing the partition sensitivity study of SOTER.
- You can also find the log files for generating this figure in
  `~/atc22-artifact/SOTER/script/sensitivity` and `~/atc22-artifact/SOTER/script/sensitivity-trans`.

**Expected results:**

- SOTER's inference latency grows as the input tensor shape increases (match **C4**).
- SOTER's inference latency decreases as the partition ratio increases in most
  cases (match **C5**).

**Important notes of Experiment 2:**

- The inference latency may fluctuate (i.e., the latency belonging to a larger partition ratio can
  be even higher). This is normal because operators are randomly selected
  to partition and execute on a GPU, and the computational overhead can be larger when such a
  partition is fragmented, i.e., the context switch overhead dominates the GPU
  performance gain (see paper $6.2).


### Experiment 3: Confidentiality study (25 mins)

This experiment runs SOTER and baseline systems on two models under model stealing
attacks to show the empirical confidentiality guarantee. The output is the test
accuracy (or BLEU score) for different systems. A larger accuracy (or BLEU score) indicates more
confidentiality loss of the serving model.

**Command to run (in the GPU terminal):**

```shell
cd ~/atc22-artifact/SOTER/script/
bash ./conf.sh
```

**Output:**
- Two pdf files named `figure7a-vgg.pdf` and `figure7b-trans.pdf` in `~/atc22-artifact/SOTER/figure/`,
  containing the stealing attack results of SOTER and baseline systems.

**Expected results:**
- In figure7a-vgg.pdf, when the partition ratio grows, SOTER achieves similar accuracy (i.e., 
  comparable confidentiality) as eNNclave
  (match **C6**).
- In figure7a-vgg.pdf and figure7b-trans.pdf, when the partition ratio grows, SOTER achieves lower accuracy (i.e., stronger
  confidentiality) than AegisDNN (match **C7**).

### Experiment 4: The pattern between SOTER's fingerprint protocol and fixed fingerprint baseline (20 mins)

This experiment runs two VGG inference programs (i.e., one for SOTER's
fingerprint protocol, one for a fixed fingerprint protocol as a
baseline), calculates and compares the l2 distance between fingerprints and
reports the pattern from an attacker's perspective.

**Command to run (in the GPU terminal):**

```shell
cd ~/atc22-artifact/SOTER/script/
bash ./run-fpcheck.sh
```

**Output:**

- Two pdf files named `figure8a-oblifp.pdf` and `figure8b-fixedfp.pdf` in `~/atc22-artifact/SOTER/figure/`,
  containing the l2 distance distribution of SOTER and baseline.
- You can also find the log files for generating figures in
  `/home/xian/atc22-artifact/SOTER/fingerprint_obliviousness/figure8a(or
  8b)/l2_dist_obli_fp.dat (or l2_dist_fixed_fp.dat)`

**Expected results:**

- SOTER's fingerprint l2 distance distribution (i.e., `figure8a-oblifp.pdf`) is close to a
  normal distribution. Note that, some fluctuations may occur as SOTER randomly
  derives fingerprints at runtime for security (match **C8**).
- Baseline's fingerprint l2 distance distribution (i.e., `figure8b-fixedfp.pdf`)
  has spikes at a small distance (x-axis) and does not share the form of normal
  distribution (match **C9**). 

**Important notes of Experiment 4:**

- The core difference for verifying this experiment is that SOTER's distance distribution shares a similar form as normal
  distribution (see next bullet point), while the baseline has an entirely different distribution and
  has spikes at a small distance.
- Since SOTER uses random scalars to produce oblivious fingerprints (the
  baseline uses fixed fingerprints), owing to
  the randomness, `figure8a-oblifp.pdf` may not exactly match the form of standard normal distribution (but should be
  similar). You can ignore it directly, or you can reproduce this figure
  by re-trying the following command (might need multiple times) and check the produced figure:
  ```shell
    cd ~/atc22-artifact/SOTER/fingerprint_obliviousness/figure8a
    bash ./runobli.sh
  ```
