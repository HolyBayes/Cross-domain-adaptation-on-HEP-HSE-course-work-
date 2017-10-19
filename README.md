

---
# Gradient reversal for MC/real data calibration

---
Artem Ryzhikov$^{1,2}$, Andrey Ustyuzhanin$^{1,2,3}$

**ACAT 2017 University of Washington,
Seattle, August 21-25, 2017**

<sub>$^1$ NRU Higher School of Economics, Russia
$^2$ Yandex School of Data Analysis
$^3$ Moscow Institute of Physics and Technology
E-mail: artemryzhikov@gmail.com
</sub>
<br></br>
### Abstract

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In the research, a new approach for finding rare events in high-energy physics was tested. As an example of physics channel the decay of $\tau \rightarrow 3 \mu$ is taken that has been published on Kaggle within LHCb-supported challenge. The training sample consists of simulated signal and real background, so the challenge is to train classifier in such way that it picks up signal/background differences and doesn’t overfits to simulation-specific features. The approach suggested is based on cross-domain adaptation using neural networks with gradient reversal **\[1\]**. The network architecture is a dense multi-branch structure. One branch is responsible for signal/background discrimination, the second branch helps to avoid overfitting on Monte-Carlo training dataset. The tests showed that this architecture is a robust a mechanism for choosing tradeoff between discrimination power and overfitting, moreover, it also improves the quality of the baseline prediction. Thus, this approach allowed us to train deep learning models without reducing the quality, which allow us to distinguish physical parameters, but do not allow us to distinguish simulated events from real ones. The third network branch helps to eliminate the correlation between classifier predictions and reconstructed mass of the decay, thereby making such approach highly viable for great variety of physics searches.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The method shows significantly better results than Data Doping **\[2\]**. Moreover, gradient reversal gives more flexibility and helps to ensure flatness of the network output wrt certain variables (e.g. nuisance parameters) as well. Thus, described approach is going to be new *state-of-the-art*.
<br></br>
### 1. Introduction

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Nowadays there are lots of machine learning approaches applied for filtering rare events in high-energy physics. Methods like Decision Trees, Linear models, Boosting and Neural Networks found a great application in such issues. Today, with the advent of new approaches of deep learning, new frameworks and greater computing power neural networks are becoming more and more relevant.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;However the application of a large number of machine learning methods for the problems of high-energy physics is difficult or even impossible. Neural networks is not exception.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The reason of that is low generalization quality and high number of physical restrictions for ML models. Besides high values of metrics (in classic ML problems) models should be physical-interpretable.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For example, it's absolutely obvious, that the model able to determine the events from only specific channel of background is not interesting for physics. The models able to determine the whole family of events with common physics are interesting much more.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In the research we tried new approach for finding rare events in high energy physics. The method based on cross-domain adaptation with gradient reversal **[1]**, and presents dense multi-branch neural network: first branch is responsible for signal detection, other branches helps to avoid overfit on Monte-Carlo and mass pick.
The given results shows that explored architecture shows best results (in comparison with Data Doping technique **[2]**). It helps to avoid overfiting without any loss of quality! Thus, described approach is going to be new *state-of-the-art*.

<br></br>
### 2. Problem

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Usage of Monte Carlo-generated sample is fairly common approach in the High Energy Physics. However it's often hard to include all physics factors into Monte-Carlo simulation. Moreover, not all variables can be simulated accurately enough, so the discrepancies may lead either to:
***a)*** expensive simulation of both signal and background, or to..
***b)*** ML models trained on simulated sample that overfits to the simulated artifacts and work poorly on the real data.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In particular such issue is actual in such problem like rare signal detection. The general reason is that it's too hard to mine enough real signals for training dataset.

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
### 3. Baseline

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;On publication moment *Data Doping* **[2]** was the best technique to train classifier on Monte-Carlo without overfit. We selected it as a baseline.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The idea of Data Doping is to “dope” the training set with a small number of Monte-Carlo events from the control channel C, but labeled as background. Thus, it helps the classifier to dissalow features discriminating real and background. The technique is shown on ***Figure 3*** below.

![](https://i.imgur.com/M7DfpHu.png)
<center><b>Figure 3.</b> Data doping</center>


The optimal number of doping events was taken from **[2]**.



<br></br>
### 4. Domain adaptation
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;As an alternative for *Data Doping* we discovered new method, based on *Cross-Domain adaptation with gradient reversal* **[1]**. The concept is the same as in *GAN (Generative Adversarial models)* **[7]**: we use an additional branch, which is trained to discriminate real from background (discriminator). But we also reverse gradient from discriminator to make general model not able to discriminate real from background. It's visualized on <b><i>Figure 4</i></b>

<br></br>
![](https://i.imgur.com/MHstIVz.png)
<center><b>Figure 4</b> Cross-domain adaptation with gradient reversal</center>

<br></br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Instead, we placed reversal into loss sign using following simple equation $$-\frac{\partial L_d}{\partial \Theta_d}=\frac{\partial-L_d}{\partial \Theta_d}$$
and used **minus** cross-entropy as a final objective for ***Domain classifier*** part (***Figure 5***)
<br></br>
![](https://i.imgur.com/Ry7EqQn.png)
<center><b>Figure 5</b> Equivalent form of gradient reversal architecture (<b><i>Figure 4</i></b>)</center>

<br></br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The network architecture has a dense 2(3*)-branch structure (<b><i>Figure 5</i></b>) and consists from following parts: 
	1. **Feature extractor** – responsible for feature generation
	2. **Label predictor** – responsible for the target prediction (signal/background discrimination)	
	3. **Domain classifier** – responsible for cross-domain adaptation and prevents the network from overfitting to MC domain
	4*. **Mass predictor** - helps to eliminate the correlation between classifier predictions and reconstructed mass of the decay


<sub>$^*$-**Mass predictor** part (branch) wasn’t tested in this research and our architecture was tested without this part. The <b><i>Figure 5</i></b> was drawn without this part. Theoretically it was designed as additional branch as domain classifier, working along the same principle</sub>

### 5. Data

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Training dataset (Analysis channel) consists of 67000+ events of signal ($τ → 3µ$) and background events. Control channel consists of 71000+ events of signal ($D_s → φπ$) and background. All events are described by 46 features. 

### 6. Training

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The architecture was implemented on ***Python 2.7*** using ***Lasagne (ver. 2.1)*** framework. We tuned the following parameters to obtain stable results:
* learning rates ratio between branches (**learning_rate_multiplier**);
*  batch sizes ratio for branches. The best observed values were 1000 and 300 for **Label predictor** and **Domain classifier** respectively
*  number of batches per epoch ratio. The best observed ratio between batches number was 6:1 for **Label predictor** and **Domain classifier** respectively 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The model was trained for 20 epoches with RMSProp optimizer. To achieve highest stability and reproducibility we trained only **Feature extractor** and **Label predictor** parts several (20) epoches with frozen **Domain classifier** (1'st step) and after that **Domain classifier** was trained (2'nd step).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In first step, as it was mentioned before, only **Feature extractor** and **Label predictor** were trained. Training procedure was 20-epoches RMSProp-optimization of categorical (2-classes) cross-entropy loss. Batch size was 1000. Learning rate at first step was 0.01 and was decayed 10 times each 5 epoches
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;On the second step we restored **Feature extractor** and **Label predictor** from previous step and trained all parts. **Feature extractor** and **Label predictor** trained with the same way (20 epoches, same loss, same batch size, same initial learning rate and the same learning rate decay policy). **Domain classifier** was trained with the same parameters in each 6 batches of **Feature extractor** and **Label predictor** training and with different batch size (batch_size=300)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To eliminate ***KS***-value we increased such Domain classifier’s parameters as learning rate, corresponding batch size and batches frequency. ***Figure 6*** represents such dependency from one of such parameters. It was observed that too small values of ***KS*** makes ***CvM*** values higher and ***AUC*** metric smaller. So the goal was to find balance between ***KS***, ***CvM*** and ***AUC*** using parameters described above.

<br></br>
![](https://imgur.com/ku9Dcuu.png)
<center><b>Figure 6.</b> Metrics dependency from domain classifier’s <b>learning_rate_multiplier</b></center>

<br></br>
### 7. Results
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In the research following models were compared: **Baseline** (label predictor from ***Figure 5*** without Domain Adaptation), **Domain Adaptation** (our approach), **Data Doping**. Models were tested on 85000+ events of signal ($τ → 3μ$) and background.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The tests showed (***Figure 7***) that this architecture is a robust mechanism for choosing tradeoff between discrimination power and overfitting, moreover, it also improves the quality of the baseline prediction. Thus, this approach allowed us to train deep learning models without reducing the quality, which allow us to distinguish physical parameters, but do not allow us to distinguish simulated events from real ones. 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;As shown in the table below our method provides the best solution for signal detection problem ($𝜏 → 3𝜇$). 


<br></br>
|                       | AUC (truncated) | KS (< 0.09) | CvM (<0.002) |
|-----------------------|-----------------|-------------|--------------|
| Mass-aware Classifier | **0.999**           | <span style="color:red">0.18</span>        | <span style="color:green">0.0008</span>       |
| Data Doping           | 0.9744          | <span style="color:green">0.087</span>       | <span style="color:green">0.0011</span>       |
| Domain-adaptation     | 0.979           | <b><span style="color:green">0.06</span></b>        | <b><span style="color:green">0.0008</span></b>       |
<center><b>Figure 7.</b> Results</center>

<br></br>
### 8. Conclusion
The method proposed is shown to work well on a typical particle physics analysis
problem:
- Remarkable classification quality;
- Ro- Robustness to MC / Real data mixture;
- Uniformity of the output wrt chosen (mass or nuisance parameter) feature
- Tradeoff between discrimination power and overfitting tuned (***Figure 6***)

<br></br>
### References

**[1]** Ganin, Y, and V. Lempitsky. ***Unsupervised domain adaptation by backpropagation.***
International Conference on Machine Learning. 2015.
**[2]** V. Gaitan. ***Data Doping solution for “Flavours in Physics” challenge***
https://indico.cern.ch/event/433556/contributions/1930582/
**[3]** ***Flavours of Physics Competition***, https://www.kaggle.com/c/flavours-of-physics
**[4]** A. Rogozhnikov, A. Bukva, V. Gligorov, A. Ustyuzhanin, M. Williams
***New approaches for boosting to uniformity***, JINST, 2015
**[5]** A. Ryzhikov, A. Ustyuzhanin ***Source code for Domain Adaptation research***
https://github.com/Leensman/Cross-domain-adaptation-on-HEP-HSE-course-work-
**[6]** A. Ryzhikov, A. Ustyuzhanin ***Gradient reversal for MC/real data calibration***
https://indico.cern.ch/event/567550/contributions/2629724/attachments/1513629/2361286/Ryzhikov_poster_v6.pdf
**[7]** Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio ***Generative Adversarial Networks***
https://arxiv.org/abs/1406.2661