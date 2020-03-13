# Experiment on Bayesian Regression (for educational purpose)
-------------------
###### Credit to this tutorial can be assigned as:
```
J. Wang, Bayes_for_Regression, (2020), GitHub repository, https://github.com/wonjohn/Bayes_for_Regression
```
Short recap
-------------------
_Bayesian epistemology_ introduces important constraints on top of rational degrees of belief and a rule of probabilistic inference--the principle of conditionalization, according to [William Talbott, 2008](https://plato.stanford.edu/entries/epistemology-bayesian/).

_Bayesian statistics_ forms a major branch in _statistics_. _Bayesian statistics_ relies on Bayesian principle to reveal a beautiful epistemology scheme through probabilistic inference: one should rationally updates degrees of knowing or belief once new evidence is observed. Mathematically, it is denoted as:

_**P(S|E) = P(E|S)P(S)/P(E)**_

where, _**s**_ can be any arbitrary statement, and _**E**_ is observed evidence(s). Without observing any evidence, it is rational to stay with idealized belief denoted as the _prior_ belief _**P(s)**_. But if we have observed an evidence, there is something we can do to update our belief. One option is to utilize the measurement called the _likelihood_ function that quantifies how our _prior_ belief should manifest the evidence at hand. The _likelihood_ function _**P(E|S)**_ together with the _prior_ function _**P(S)**_ help to update our belief once there is more information from the reality. The updated belief is called the _posterior_ function of _**S**_, which is _**P(S|E)**_.

In this small snippet of experiment, the principle of _Bayesian statistics_ is showcased through a prevalent prediction problem: _regression_.

Showcase
-------------------
### * Linear regression formalization
In the simplest case of _univariate linear regression_, the ultimate goal is to find out the true model with the highest confidence so that we can make confident predictions by using the model. 

Without considerting the confidence, conventional linear regression only achieves point estimation of the model parameter through _least squares_ method, which is not a bad idea as we could be able to obtain exact prediction by using the model. But in reality, we also would like to know how certain it can be for predition with the model at hand. In many practical cases, such as predicting housing values, stock values, pollution concentration, soil mineral distribution, etc., apart from the estimation, we also prefer to see the confidence of our model performance so as to control the risk of making the prediction or any possible economic loss. Such preference leads to natural transition from point estimation of the model parameter to a probabilistic perpective, and paves the way to the application of _Bayesian statistics_.

### * Bayesian inference
Take the following _univariate linear regression_ problem for example, _Bayesian statistics_ attempt to find out the true model and with quantified confidence simultaneously. Visually, _Bayesian statistics_ tries to figure out the true form of the unobserved linear model (linear parameters) in Fig.1 through few point observations (points in this example).

<img src="/img/0_data.png" width="450" heigth="390"> 

_Fig.1 Linear regression problem setting._

Here in Fig.1 I intentionally reveal the true linear function as a line so that we can compare how _Bayesian statistics_ help us to recover the true model only from limited observations. The true _univariate linear_ model I used here is:

_**M(x) = 3 + 2x**_

Mathematically, without knowing the true model parameters _**3**_(for the interception) and _**2**_(for the slope), Bayesian statistics_ would quantify _θ<sub>1</sub>_ and _θ<sub>12</sub>_ for the model:

_**M(x) = θ<sub>1</sub> + θ<sub>2</sub>x**_

or in a vectorized form:

_**M(X) = θ<sup>T</sup>X**_

where _**X**_ is _[1, x]<sup>T</sup>_ and _**θ**_ is _[θ<sub>1</sub>, θ<sub>2</sub>]<sup>T</sup>_. Unfortunately, like I said, in most cases, we only have noisy observations _**Y**_ such as _**(y<sub>1</sub>, y<sub>2</sub>, ... , y<sub>n</sub>)**_ with preferable _Gaussian_ distributed noises _**ε**_ around the true model, in the form of:

_**Y = θ<sup>T</sup>X + ε**_

As being seen in Fig.1, there are 3 observations available for us to find out the model parameters _**θ**_. According to the _Bayesian principle_, the inference of the model parameter can be achieved through:

_**P(θ|D) = P(D|θ)P(θ)/P(D)**_

where _**D**_ is a collection of observed noisy point pairs _**{(x<sub>1</sub>, y<sub>1</sub>), (x<sub>2</sub>, y<sub>2</sub>), (x<sub>3</sub>, y<sub>3</sub>)}**_ as shown in Fig.1. In order to quantify the _posterior_ _**P(θ|D)**_, the problem now reduces to specify the _likelihood_ _**P(D|θ)**_ and _prior_ _**P(D)**_ as mentioned above, which is a 2-step process. Although, specifying both functions is non-trivial, in this simple experiment, we stay simple to keep this snippet with the simplest case.

### * Bayesian function specification: _Likelihood_
For any single observed data point _**(x<sub>k</sub>, y<sub>k</sub>)**_, the _likelihood_ measures the probability of the model parameter _**θ**_ gives rise to this known data point. Thus, _given_ any possible _**θ**_, how likely it is to observe this particular point of tuple _**(x<sub>k</sub> , y<sub>k</sub>)**_? Referring above to the noisy observation from the linear model, by saying we have observations with noise _**ε**_ around the true model, it is most handy to impose a _Gaussian_ distribution over the noise around the true model. In short, the _likelihood_ of observing the tuple _**(x<sub>k</sub> , y<sub>k</sub>)**_ follows a _Gaussian_ distribution around the model specified by _**θ**_:

_**P(D|θ) = P((x<sub>k</sub> , y<sub>k</sub>)|θ) ~ N(y<sub>k</sub> ; θ<sup>T</sup>X, ε)**_

This _Gaussian_ form _likelihood_ can be easily implemented as a function in _python_ as:

```python
def likeli(theta1,theta2,obs_y,obs_x):  # It is a function of theta with known observations
    sigma = 1  # Standard deviation of the Gaussian likelihood
    func = (1/np.sqrt(6.28*sigma**2))*np.exp((obs_y-theta1-theta2*obs_x)**2/(-2*sigma**2))
    return func
```
where I chose a standard deviation of _**1**_ for this _Gaussian likelihood_ as I assume for the noise level. It shows my confidence interval that the observations should be in around the true linear model. This _likelihood_ is obviously a function wrt. _**θ**_ as the tuple _**(x<sub>k</sub> , y<sub>k</sub>)**_ is observed. More intuitively, if we observe one pair of _**(x<sub>k</sub> , y<sub>k</sub>)**_ as denoted red to the left of Fig.2, the above _likelihood_ is a function of _**θ**_ or _[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>_, and can be plotted in a 2-dimensional space defined by θ<sub>1</sub> and θ<sub>2</sub> to the right of Fig.2. Here are two separate observations visualized, each of whose _likelihood_ function is plotted and appears roughly to be a line as a function of _**θ**_. It is only roughly a line in the space of _[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>_, implying that there are infinite number of _[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>_ options running all the way from positive to negative values to give rise to the observation. This is extremely reasonable as one point observation determines lines with either positive or negative interception and slope. Continue with the sample linear function above, if we can be able to make a couple of more noisy observations, we can obtain multiple _likelihood_ for each of the observation in the _[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>_ space.

<img src="/img/1_likeli_2.gif" width="800" heigth="680"> 

_Fig.2 Likelihood wrt. a single observation._

In this case of linear regression, isn't it getting clear that these line-shaped _likelihood_ functions are potentially intersected at some relatively fixed region? That is where we can combine these _likelihood_ function that the profile of _**θ**_ can be delineated. In what way to combine? As the _likelihood_ is a probability measurement, combining the _likelihood_ is simply a joint probability. Observing each data point as a _**(x<sub>k</sub> , y<sub>k</sub>)**_ tuple is considered to be an [_**iid**_](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) process, thus the joint _likelihood_ of any _**θ**_ gives rise to all the observations is a multiplication of all the individual _likelihood_:

_**P(D|θ) = ∏<sub>i</sub> P((x<sub>i</sub> , y<sub>i</sub>)|θ)**_

The animation (Fig.3) below shows how this joint probability is updated with each added observation. It is quite appealing that when the second _**(x<sub>k</sub> , y<sub>k</sub>)**_ tuple is observed, the joint _likelihood_ function already started take in shape and the inference of the model parameter can be made in a well delineated subspace within the _[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>_ space. The joint _likelihood_ with an ellipse-shaped _Gaussian_ centers around _**[3, 2]**_ indicating a high confidence that it should be the model parameter. At the same time, the _likelihood_ does not reject other possibilities as it is still possible, with a certain noise level in the observations, that _[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>_ can take some other values around _**[3, 2]**_. When the third point is observed, this _likelihood_ gives a more shrinked distribution representing the process of knowledge update wrt. _[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>_.

<img src="/img/1_likeli1.gif" width="800" heigth="680"> 

_Fig.3 Joint likelihood wrt. to observations._

At this point, it is no suprising that why the [_**Maximum Likelihood Estimation (MLE)**_](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) is so frequently adopted. For linear regression, especially simple as there is only few independent variable and enough observations, and of course without too much noise, the joint _likelihood_ function could already bring a desirable results. And it is then also quite safe to maximize the _likelihood_ to obtain a point estimation of the model parameter. 

### * Bayesian function specification: _prior_

Different from that _likelihood_ can be delineated from observed information, choosing _prior_ function for _Bayesian_ inference is tricky. It is called _prior_ distribution because it requires us to configure the distribution of the model parameters _prior_ to seeing any data, or based upon our _prior_ knowledge with regard to the parameters. Fortunately, in several situation, we DO have such prior knowledge when building a regression model. For instance, if someone is interested in the loss of a particular type of soil _**(y<sub>k</sub>)**_ due to rainfall _**(x<sub>k</sub>)**_ in a region, it is already handy to know that there should be a positive relationship between _**(y<sub>k</sub>)**_ and _**(x<sub>k</sub>)**_. It also means that we can more or less _constrain_ the _**θ**_ to be positive. Or, maybe someone has already done similar work in other places and brought some confident results, it is even possible to further _constrain_ the _**θ**_ to be a probabilistic distribution over these available results (values).

But here in this experiment, although it is a simple linear regression example, we are running into a awkward situation: nothing is availabe except the observations to make inference about the model. This is where one has to rely on _improper_ or _non-informative_ _prior_ distribution for the model parameters. These keywords such as _improper_ and _non-informative_ indicate that the design of the _prior_ function is entirely arbitrary. One option is to make _**P(θ)=1**_, thus it is _non-informative_ and will have no effect over the _posterior_ when multplies with the _likelihood_. Another option is to make a general assumption that _**P(θ)**_ follows some well formed statistical distribution, such as _Gaussian_ distribution shown in Fig.4 below. This kind of specification can be _improper_ as it would potentially impose limited and unreasonable assumption that _**θ**_ is normally distributed around _**0**_.

<img src="/img/2_prior.png" width="380" heigth="380"> 

_Fig.4 Non-informative prior distribution for model parameters._

This _improper Gaussian_ distribution within the _[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>_ space means that if we draw points randomly from it, it is more likely to have _**0**_ with values close to zero. Visually, each random point in the _[θ<sub>1</sub> , θ<sub>2</sub>]<sup>T</sup>_ space determines a random line in the space of _[x, y]<sup>T</sup>_ as shown below in Fig.5, but most lines are with zero interception and slope.  

<img src="/img/2_priorDraw.gif" width="800" heigth="680"> 

_Fig.5 Randomly drawn prior functions for the model._

The effect of this _improper Gaussian prior_ combined with the _likelihood_ can be visualized as in Fig.6. The combination, again, follows the principle of joint statistical distribution, is achieved through multiplication. The resultant _posterior_ distribution of the model parameter _**0**_ is compared with their _likelihood_ distribution solely determined by observed evidences. 

<img src="/img/3_post_likeli.gif" width="330" heigth="330"> 

_Fig.6 Posterior distribution/function for model parameters._

In _Bayesian statistics_, this multiplication is normally referred as _update_ as mentioned at the beginning of this experiment, where the _prior_ knowledge is _updated_ by the _likelihood_ brought by observations. Equivalently, I could also say that the _likelihood_ is being _shifted_ or _dragged_ by our _prior_ blief, because our _prior_ blief imposes a constraint even we have observed few evidence. One can see that the direction of the _shifting likelihood_ is towards the _prior_. But as it is a multiplication, the _shift_ would stay around the large value in both _prior_ and _likelihood_, thus the _shift_ is also along the gentle gradient of the _likelihood_ while moving towards the _prior_.

The _improperness_ of the _improper prior_ in this case is nicely highlighted as we already know that the _likelihood_ is perfectly centered around the true parameter values of _**[3, 2]**_, and now it is _shifted_ away. But we would never notice this in practice as we wouldn't know the true model parameters.

Now, one may start to ask: what on earth is the point to use these badly designed _prior_ functions?! Unfortunately, there is no anwser. There is even no scientific guidelines for choosing a _prior_. It is the nature of statistic inference that we are force to make some assumptions from scratch, just like we have to assume there is a linear relationship already as we only see few data points. There are [few discussions](https://stats.stackexchange.com/questions/27813/what-is-the-point-of-non-informative-priors) going around that one can refer. More formal research can be found in [_**The Bayesian Choice: From Decision-Theoretic Foundations to Computational Implementation (Springer Texts in Statistics**_](https://www.amazon.com/dp/0387715983/), as well as [_**Moving beyond noninformative priors: why and how to choose weakly informative priors in Bayesian analyses**_](https://onlinelibrary.wiley.com/doi/10.1111/oik.05985).

### * Bayesian posterior
- More observations would wash away the effect of the _improper_ or _non-informative prior_ assumption.



### * Remarks
- MLE can be enough
- model selection vs. likelihood specification: likelihood in essence only specifies model parameter distribution
- great property of Gaussian: multiply
- informative vs. non-informative prior
- Bayesian framework: ecapsulate many special cases such as _least squares_ when Gaussian is assumed, and prior relates to regularization
- function space vs. parameter space
- feature space discussion

