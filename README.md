

---
# Gradient reversal for MC/real data calibration

---

Artem Ryzhikov$^{1,2}$, Andrey Ustyuzhanin$^{1,2,3}$

**ACAT 2017 University of Washington,
Seattle, August 21-25, 2017**

<sub>$^1$ Yandex School of Data Analysis
$^2$ NRU Higher School of Economics
$^3$ Moscow Institute of Physics and Technology
E-mail: artemryzhikov@gmail.com, andrey.ustyuzhanin@cern.ch
</sub>
<br></br>
### Abstract

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In this research we propose a technique for training neural networks on mixture of MC-simulated signal and real background sample that allows to avoiding overfitting to simulated artifacts. The technique is based on cross-domain adaptation approach with gradient reversal **\[1\]**. The method shows significantly better results than Data Doping **\[2\]**. Moreover, gradient reversal gives more flexibility and helps to ensure flatness of the network output wrt certain variables (e.g. nuisance parameters) as well.
<br></br>
### 1. Introduction

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Usage of Monte Carlo-generated sample is fairly common approach in the High Energy Physics. However, not all variables can be simulated accurately enough, so the discrepancies may lead either to:
***a)*** expensive simulation of both signal and background, or to..
***b)*** ML models trained on simulated sample that overfits to the simulated artifacts and work poorly on the real data.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In the research we propose a technique to train neural networks that are safe to train on mixture of simulated and real data.
<br></br>
### 2. Problem
<br></br>
![](https://i.imgur.com/JTn5NTy.png)
<center> <b>Figure 1.</b> Training on the mixture of simulated (MC) and real data 
</center>
<br></br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In the research we use *τ → 3µ* events as signal (analysis) channel that has been published at the Data Science [challenge](https://www.kaggle.com/c/flavours-of-physics) on kaggle.com **[3]**. The challenge is three-fold:
1) Since the classifier is trained on a mixture of simulated signal and real data background, it is possible to reach a high performance by exploiting features that are not perfectly modeled in the simulation. We require that the classifier should not have a large discrepancy when applied to real and simulated data. To verify this, we use a control channel, $D_s → φπ$, that has a similar topology as the signal decay, $τ → 3µ$ (analysis channel). $D_s → φπ$ is a well-known decay, as it happens much more frequently. So the goal is to train a classifier able to separate A from B but not C from D (<b><i>Figure 1</i></b>). A Kolmogorov-Smirnov (KS) test is used to evaluate the differences between the classifier distribution on each sample. In our problem KS is calculated between prediction’s distributions for real and simulated data for $Ds → φπ$ channel. The KS-value of the test should be less than 0.09.
2) The classifier output should not be correlated with reconstructed mass feature, i.e. it’s output distribution should not sculpt artificial bumps that could be interpreted as a (false) signal. To test the flatness we’ve used Cramer-von Mises (CvM) test that gives the uniformity of the distribution **[4]**.
3) The quality of signal discrimination should be as much as possible. The evaluation metric for signal discrimination is Weighted Area Under the ROC Curve (*truncated AUC*) **[3]**

<br></br>
![](https://i.imgur.com/X9Dh5Sf.png)
<center> <b>Figure 2.</b> Illustration of the CvM correlation test <b>[4]</b>. On the left side there is no correlation with mass (small CvM values). On the right side model’s predictions are highly correlated with mass (high CvM values)
</center>

<br></br>
### 3. Data Doping (Baseline)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In the research we have selected Data Doping **[2]** as a baseline. The idea is to “dope” the training set with a small number of Monte-Carlo events from the control channel C, but labeled as background. The optimal number of doping events was taken from **[2]**.

<br></br>
### 4. Domain adaptation

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The network architecture has a dense 2(3*)-branch structure (<b><i>Figure 3</i></b>) and consists from following parts: 
	1. **Feature extractor** – is responsible for feature generation
	2. **Label predictor** – is responsible for the target prediction (signal/background discrimination)	
	3. **Domain classifier** – is responsible for cross-domain adaptation and prevents the network from overfitting to MC domain
	4*. **Mass predictor** - helps to eliminate the correlation between classifier predictions and reconstructed mass of the decay

<br></br>
![](https://imgur.com/ZcP3Lmk.png)
<center><b>Figure 3.</b> Domain Adaptation
</center>
<br></br>

<sub>$^*$-**Mass predictor** part (branch) wasn’t tested in this research and our architecture was tested without this part. The <b><i>Figure 3</i></b> was draws without this part. Theoretically it was designed as additional branch as domain classifier, working along the same principle</sub>

### 5. Experiment

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Training dataset (Analysis channel) consists of 67000+ events of signal ($τ → 3µ$) and background events. Control channel consists of 71000+ events of signal ($D_s → φπ$) and background. All events are described by 46 features. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The architecture above was implemented on ***Python 2.7*** using ***Lasagne (ver. 2.1)*** framework. We tuned the following parameters to obtain stable results:
* learning rates ratio between branches (**learning_rate_multiplier**);
*  batch sizes ratio for branches. The best observed values were 1000 and 300 for **Label predictor** and **Domain classifier** respectively
*  number of batches per epoch ratio. The best observed ratio between batches number was 6:1 for **Label predictor** and **Domain classifier** respectively 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The model was trained for 20 epochs with RMSProp optimizer. 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;***KS***-value was eliminated by increasing of Domain classifier’s learning rate, increasing corresponding batch size and batches frequency. But too small values of ***KS*** makes ***CvM*** values higher and ***AUC*** metric smaller. ***Figure 4*** represents such dependency from one of such parameters. So the goal was to find balance between ***KS***, ***CvM*** and ***AUC*** using parameters described above.

<br></br>
![](https://imgur.com/ku9Dcuu.png)
<center><b>Figure 4.</b> Metrics dependency from domain classifier’s <b>learning_rate_multiplier</b></center>

<br></br>
### 6. Results
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In the research following models were compared: **Baseline** (label predictor from ***Figure 3*** without Domain Adaptation), **Domain Adaptation** (our approach), **Data Doping**. Models were tested on 85000+ events of signal ($τ → 3μ$) and background.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The tests showed that this architecture is a robust mechanism for choosing tradeoff between discrimination power and overfitting, moreover, it also improves the quality of the baseline prediction. Thus, this approach allowed us to train deep learning models without reducing the quality, which allow us to distinguish physical parameters, but do not allow us to distinguish simulated events from real ones. 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;As shown in the table below our method provides the best solution for signal detection problem ($𝜏 → 3𝜇$). 


<br></br>
|                       | AUC (truncated) | KS (< 0.09) | CvM (<0.002) |
|-----------------------|-----------------|-------------|--------------|
| Mass-aware Classifier | **0.999**           | <span style="color:red">0.18</span>        | <span style="color:green">0.0008</span>       |
| Data Doping           | 0.9744          | <span style="color:green">0.087</span>       | <span style="color:green">0.0011</span>       |
| Domain-adaptation     | 0.979           | <b><span style="color:green">0.06</span></b>        | <b><span style="color:green">0.0008</span></b>       |


<br></br>
### 7. Conclusion
The method proposed is shown to work well on a typical particle physics analysis
problem:
- Remarkable classification quality;
- Robustness to MC / Real data mixture;
- Uniformity of the output wrt chosen (mass or nuisance parameter) feature
- Tradeoff between discrimination power and overfitting tuned (***Figure 4***)

<br></br>
### References

**[1]** Ganin, Y, and V. Lempitsky. "Unsupervised domain adaptation by backpropagation."
International Conference on Machine Learning. 2015.
**[2]** V. Gaitan, Data Doping solution for “Flavours in Physics” challenge
https://indico.cern.ch/event/433556/contributions/1930582/
**[3]** Flavours of Physics Competition, https://www.kaggle.com/c/flavours-of-physics
**[4]** A. Rogozhnikov, A. Bukva, V. Gligorov, A. Ustyuzhanin, M. Williams
“New approaches for boosting to uniformity”, JINST, 2015
**[5]** A. Ryzhikov, A. Ustyuzhanin Source code for Domain Adaptation research
https://github.com/Leensman/Cross-domain-adaptation-on-HEP-HSE-course-work-
**[6]** A. Ryzhikov, A. Ustyuzhanin Gradient reversal for MC/real data calibration
https://indico.cern.ch/event/567550/contributions/2629724/attachments/1513629/2361286/Ryzhikov_poster_v6.pdf