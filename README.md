# Experiment on Bayesian Regression (for educational purpose)
-------------------

Short recap
-------------------
_Bayesian epistemology_ introduces important constraints on top of rational degrees of belief and a rule of probabilistic inference--the principle of conditionalization, according to [William Talbott, 2008](https://plato.stanford.edu/entries/epistemology-bayesian/).

_Bayesian statistics_ forms a major branch in _statistics_. _Bayesian statistics_ relies on Bayesian principle to reveal a beautiful epistemology scheme through probabilistic inference: one should rationally updates degrees of knowing or belief once new evidence is observed. Mathematically, it is denoted as:

_**P(S|E) = P(E|S)P(S)/P(E)**_

where, _**s**_ can be any arbitrary statement, and _**E**_ is observed evidence(s). Without observing any evidence, it is rational to stay with idealized belief denoted as the _prior_ belief _**P(s)**_. But if we have observed an evidence, there is something we can do to update our belief. One option is to utilize the measurement called the _likelihood_ function that quantifies how our _prior_ belief should manifest the evidence at hand. The _likelihood_ function _**P(E|S)**_ together with the _prior_ function _**P(S)**_ help to update our belief once there is more information from the reality. The updated belief is called the _posterior_ function of _**S**_, which is _**P(S|E)**_.

In this small snippet of experiment, the principle of _Bayesian statistics_ is showcased through a prevalent prediction problem: _regression_.

Showcase
-------------------
### Linear regression formalization
In the simplest case of _univariate linear regression_, the ultimate goal is to find out the true model with the highest confidence so that we can make confident predictions by using the model. Find out the true model and with quantified confidence can be achieved simultaneously through _Bayesian statistics_, which is appealing. Visually, _Bayesian statistics_ tries to figure out the true form of the unobserved linear model (linear parameters) in Fig.1 through few point observations (points in this example).

<img src="/img/0_data.png" width="340" heigth="290"> 

_Fig.1 Linear regression rationale._

Mathematically, _Bayesian statistics_ would quantify _θ<sub>1</sub>_ and _θ<sub>12</sub>_ for the model:

_**M(x) = θ<sub>1</sub> + θ<sub>2</sub>x**_

or in a vectorized form:

_**M(X) = θ<sup>T</sup>X**_

where _**X**_ is _[1, x]<sup>T</sup>_ and _**θ**_ is _[θ<sub>1</sub>, θ<sub>2</sub>]<sup>T</sup>_. Unfortunately, in most cases, we only have noisy observations _**Y**_ such as _**(y<sub>1</sub>, y<sub>2</sub>, ... , y<sub>n</sub>)**_ with preferable _Gaussian_ distributed noises _**ε**_ around the true model, in the form of:

_**Y = θ<sup>T</sup>X + ε**_

Conventional linear regression only achieves point estimation of the model parameter through _least squares_ method. _Bayesian statistics_, instead, attempts to estimate the parameters from a probabilistic perpective.

### Bayesian inference

According to the _Bayesian principle_, the inference of the model parameter can be achieved through:

_**P(θ|D) = P(D|θ)P(θ)/P(D)**_

where _**D**_ is a collection of observed noisy point pairs _**{(x<sub>1</sub>, y<sub>1</sub>), (x<sub>2</sub>, y<sub>2</sub>), (x<sub>3</sub>, y<sub>3</sub>)}**_ as shown in Fig.1. In order to quantify the _posterior_ _**P(θ|D)**_, the problem now reduces to specify the _likelihood_ _**P(D|θ)**_ and _prior_ _**P(D)**_ as mentioned above, which is a 2-step process. Although, specifying both functions is non-trivial, in this simple experiment, we stay simple to keep this snippet with the simplest case.

For the _likelihood_ function, by saying the noisy observations are _Gaussian_ distributed around the true model, it is then reasonable to assume that observations are likely to follow a _Gaussian_ distribution _given_ any known _**θ**_. Thus for any single observed data point _**(x<sub>k</sub>, y<sub>k</sub>)**_, the _likelihood_ measures the probability of the model parameter _**θ**_ gives rise to this known data point. In short, given any _**θ**_, the probability of observing this particular point of tuple _**(x<sub>k</sub> , y<sub>k</sub>)**_ is:

_**P(D|θ) = P((x<sub>k</sub>, y<sub>k</sub>)|θ)**_

Observing data is always considered to be an [_**iid**_](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) process, thus the joint _likelihood_ of any _**θ**_ gives rise to all the observations is simply a multiplication of all the individual _likelihood_:

_**P(D|θ) = ∏<sub>i</sub>P((x<sub>i</sub>, y<sub>i</sub>)|θ)**_




