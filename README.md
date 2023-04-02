174 Stock Analysis Project
================

# Comparing Amazon, Microsoft, Nvidia, and Google stocks

#### 174 Project

#### By: Jae Yun

#### Introduction:

The stock market is one of the most fascinating fields, as many stocks
within the same industry follow the same price pattern. This is
especially true in the technology industry where companies like as
Amazon, Microsoft, Nvidia, and Google have become dominant companies, as
they hold some of the largest market caps in the world. These companies
have all experienced significant growth over the past decade, and the
technology has outgrown all other industries and became the leading
sector in the past few decades.

It is interesting to study these stocks in the industry because of the
fact that we can determine the entire technology industry using just
these four stocks. However, the main idea is that we can to make
forecasts using the stocks and determine if each of the stocks are
related to their technological counterparts, as it is a crucial
component of a stock market analyst in predicting the future of the
stock. While analyzing the technology industry as a whole can provide
valuable insights into overall trends and economic conditions,
predicting the performance of individual stocks can help investors and
analysts make more informed decisions about their specific investments,
especially ones as stable as Amzn, Msft, Nvda, and Googl.

While many companies prefer to diversify their stock portfolio, some
companies tend invest in individual stocks from a certain industry as a
way to generate higher returns, especially if they have the knowledge
about the certain company they are investing in. This strategy is a
high-reward type situation that would help the company thrive but also
pose high risks to due the volatility of individual stocks and their
strong reactions from current market trends. However, the time period
from 2010 to 2019 was a relatively stable growth period in terms of
political and economical stability, and thus, companies would have
chosen a riskier approach during this decade.

Thus, we play the role of a data analyst in a company following a
riskier investing scheme, with goal of utilizing each individual stock
to predict their future performance and see which of the four stocks
have a higher return of investment within the next year between the
beginning of 2019 to end of 2019 based on the data from 2010 to the end
of 2018. We want to see which of the four stocks will have the highest
predicted performance of the 2019 year, in hopes that we can use the
same model to predict the 2020 year.

For our project, we will use four weekly prices of Amazon, Microsoft,
NVIDIA, and Google from the 10 year time frame between January 1st,
2010, to January 1st, 2020, and compare the datasets to each other. Our
datasets were retrieved from Yahoo Finance, with 522 observations
representing each week’s price opening for each of the four tech stocks.

We wanted to find a certain time frame in which a major event did not
occur, so that we have a more reliable prediction that is based on the
trend and seasonality of our data throughout the years, and rather not
based on certain events, such as the coronavirus pandemic, the
Russo-Ukrainian war, or the 2008 recession. Albeit the time frame from
2010 to 2020 had some events. This is a biased dataset because we have
knowledge of the past and the events that have occured, but we want to
consider our model forecasts if we hypothetically did not have any major
stock-changing events since we want a more technical analysis/forecasts
of the stock rather than having to consider actual events, which we
can’t really quantify in our time series predictions.

## FORMULAS USED:

#### Augmented Dickey-Fuller (ADF Test):

$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \delta_1 \Delta y_{t-1} + \cdots + \delta_{p-1} \Delta y_{t-p+1} + \varepsilon_t$$

where $\Delta y_t$ is the first difference of the time series at time
$t$, $\alpha$ is a constant term, $\beta$ is the coefficient of a linear
time trend, and $\gamma$ is the coefficient of the lagged value
$y_{t-1}$. The terms $\delta_i$ are the coefficients of the lagged first
differences, up to a maximum lag of $p-1$. The error term
$\varepsilon_t$ is assumed to be white noise.

#### Decomposition: Y(t) = T(t) + S(t) + R(t) where:

\*Y(t) is the observed value of the time series at time t = T(t) +
S(t) + R(t)

\*T(t) is the trend component of the time series at time t $T_t = f(t)$

*S(t) is the seasonal component of the time series at time
*$S_t = \sum_{i=1}^n s_i$

*R(t) is the residual or random component of the time series at time t
*$R_t = Y_t - T_t - S_t$

#### ACF:

$$\text{ACF}(k) = \frac{\sum_{t=k+1}^n (Y_t - \bar{Y})(Y_{t-k} - \bar{Y})}{\sum_{t=1}^n (Y_t - \bar{Y})^2}$$

where $k$ is the lag, $n$ is the number of observations in the time
series, $Y_t$ is the observed value at time $t$, and $\bar{Y}$ is the
sample mean of the time series.

#### PACF:

$$\rho_0 = 1$$

$$\rho_k = \frac{\text{cov}(Y_t, Y_{t-k})}{\text{var}(Y_t)} - \sum_{j=1}^{k-1} \phi_j \rho_{k-j}$$

where $\rho_k$ is the partial autocorrelation at lag $k$,
$\text{cov}(Y_t, Y_{t-k})$ is the covariance between $Y_t$ and
$Y_{t-k}$, $\text{var}(Y_t)$ is the variance of $Y_t$, $\phi_j$ is the
estimated coefficient of the autoregressive (AR) model at lag $j$, and
$k$ is the maximum lag for which the PACF is computed.

#### Differencing:

$y't = y_t - y{t-52}$ For our 52 weekly period differencing

$y't = y_t - y{t-1}$ For taking the first difference

where $y_t$ is the observed value of the time series at time $t$, and
$y'_{t}$ is the differenced value of the time series at time $t$.

#### Sarima Model:

$$(1 - \phi_1 B - \cdots - \phi_p B^p)(1 - \Phi_1 B^s - \cdots - \Phi_P B^{sP})^d (y_t - \mu)$$
$$= (1 + \theta_1 B + \cdots + \theta_q B^q)(1 + \Theta_1 B^s + \cdots + \Theta_Q B^{sQ})\varepsilon_t$$
where $y_t$ is the observed value of the time series at time $t$, $\mu$
is the mean of the time series, $\varepsilon_t$ is the error term at
time $t$, and $B$ is the backshift operator. The parameters $p$, $d$,
and $q$ correspond to the order of the autoregressive (AR), integrated
(I), and moving average (MA) components, respectively. The parameters
$P$, $D$, and $Q$ correspond to the order of the seasonal AR, seasonal
I, and seasonal MA components, respectively. The parameter $s$
represents the length of the seasonal cycle.

The $\phi$ and $\theta$ parameters are the coefficients of the AR and MA
terms, respectively, while the $\Phi$ and $\Theta$ parameters are the
coefficients of the seasonal AR and seasonal MA terms

#### Ljung-Box Test

$$Q = n(n+2) \sum_{k=1}^h \frac{\hat{\rho}_k^2}{n-k}$$

where $Q$ is the test statistic, $n$ is the sample size, $h$ is the
number of lags being tested, and $\hat{\rho}_k$ is the sample
autocorrelation at lag $k$. The null hypothesis of the test is that the
autocorrelations up to lag $h$ are equal to zero, indicating that the
time series is a white noise process. The alternative hypothesis is that
the autocorrelations are not equal to zero, indicating the presence of
serial correlation in the data. The test statistic $Q$ follows a
chi-squared distribution with $h-p$ degrees of freedom, where $p$ is the
number of parameters estimated in the time series model.

#### Box-Pierce Test

$$Q^* = n \sum_{k=1}^h \hat{\rho}_k^2$$

where $Q^*$ is the test statistic, $n$ is the sample size, $h$ is the
number of lags being tested, and $\hat{\rho}_k$ is the sample
autocorrelation at lag $k$. The null hypothesis of the test is that the
autocorrelations up to lag $h$ are equal to zero, indicating that the
time series is a white noise process. The alternative hypothesis is that
the autocorrelations are not equal to zero, indicating the presence of
serial correlation in the data.

The test statistic $Q^*$ follows an approximately chi-squared
distribution with $h-p$ degrees of freedom, where $p$ is the number of
parameters estimated in the time series model.

## Data Analysis:

The 5 V’s of our Data: Volume: We have a significant amount of data,
ranging from weekly data for a span of 10 years in the stock market.
Thus, we can use this vast amount of data to make our predictions more
effectively Velocity: Our velocity of our data is the weekly aspect
Variety: Albeit having 4 of some of the largest stocks in the technology
industry, we have a variety in the sense that these prices are all
mutually exclusive to their own stock, and each company has different
behaviors. Veracity: Since we are working with data retrieved from Yahoo
Finance, we have pinpointed accuracy in our data because these prices
were reflected on the actual stock market prices during these times. We
have a very strong accuracy that our data is true. Value: By analyzing
this data, we can determine trends and make analysis of the data as well
as look into the future of how the data will perform, and also use this
forecasting to predict other aspects of the stock market or economy,
since these powerhouse stocks are highly impactful of the tech industry.

Now, we can begin importing our data from yahoo, read csv, and plot the
time series data:

Print summary of NVDA stock as reference

``` r
print(summary(NVDA_Weekly))
```

    ##       Date                 Open             High             Low        
    ##  Min.   :2010-01-01   Min.   : 2.265   Min.   : 2.365   Min.   : 2.163  
    ##  1st Qu.:2012-06-30   1st Qu.: 3.720   1st Qu.: 3.850   1st Qu.: 3.576  
    ##  Median :2014-12-29   Median : 5.155   Median : 5.268   Median : 4.999  
    ##  Mean   :2014-12-29   Mean   :17.746   Mean   :18.425   Mean   :17.070  
    ##  3rd Qu.:2017-06-28   3rd Qu.:35.491   3rd Qu.:36.924   3rd Qu.:33.289  
    ##  Max.   :2019-12-27   Max.   :69.573   Max.   :73.190   Max.   :67.900  
    ##      Close          Adj Close          Volume         
    ##  Min.   : 2.240   Min.   : 2.055   Min.   :4.140e+07  
    ##  1st Qu.: 3.693   1st Qu.: 3.416   1st Qu.:1.517e+08  
    ##  Median : 5.152   Median : 4.949   Median :2.195e+08  
    ##  Mean   :17.828   Mean   :17.529   Mean   :2.516e+08  
    ##  3rd Qu.:35.921   3rd Qu.:35.609   3rd Qu.:3.127e+08  
    ##  Max.   :69.823   Max.   :69.230   Max.   :1.317e+09

Check the class of our dataset.

``` r
class(NVDA_Weekly)
```

    ## [1] "spec_tbl_df" "tbl_df"      "tbl"         "data.frame"

As shown above, we have a tibble dataframe of our stocks we read in from
our 4 csv files of the NVDA, GOOGL, MSFT, and AMZN stocks, with column
values corresponding to the dates price at stock open, close, as well as
the low and high prices during open, and finally, the volume. Our
dataset ranges from the beginning 2010 to the end of 2019, calculating
the weekly open prices for each week.

``` r
GOOGL_Weekly_Open_ts <- ts(GOOGL_Weekly$Open, start = c(2010, 1), frequency = 52)
NVDA_Weekly_Open_ts <- ts(NVDA_Weekly$Open, start = c(2010, 1), frequency = 52)
MSFT_Weekly_Open_ts <- ts(MSFT_Weekly$Open, start = c(2010, 1), frequency = 52)
AMZN_Weekly_Open_ts <- ts(AMZN_Weekly$Open, start = c(2010, 1), frequency = 52)
```

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

Our tech stocks have in common a steady increase weekly from 2010 to
2020.

Checking for NA values in our dataset:

    ##   googl_isna nvda_isna msft_isna amzn_isna
    ## 1          0         0         0         0

Constructing histograms:

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

As we see from our histograms, we have highly skewed data and requiring
a transformation to normalize.

Use the Augmented Dickey-Fuller Test to check for stationarity in each
stock

    ##   GOOGL_ADF_P_VALUE NVDA_ADF_P_VALUE MSFT_ADF_P_VALUE AMZN_ADF_P_VALUE
    ## 1         0.1877035         0.656367        0.8858777        0.6416679

Using our adf tests, we can tell that all of our stocks are
non-stationary processes, since they are above the significance value (p
value \> 0.05) and thus we fail to reject the non-stationarity null
hypothesis.

## Transforming our Data to a Stationary Process

We will transform our dataset to remove our trend and seasonal
components by performing a decomposition and extracting its residuals.
First, we will get the first difference of our datasets because our data
is non-stationary, and we need to remove our trends and seasonality. As
we can see from our decomposition graphs, we clearly have a trend and
seasonal aspect of our dataset:

``` r
# Google's Decomposition plot
plot(decompose(gtr))
```

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

``` r
# Nvidia's Decomposition plot
plot(decompose(ntr))
```

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-10-2.png)<!-- -->

``` r
# Microsoft's Decomposition plot
plot(decompose(mtr))
```

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-10-3.png)<!-- -->

``` r
# Amazon's Decomposition plot
plot(decompose(atr))
```

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-10-4.png)<!-- -->

We can see that we have a clear trend and seasonality from our
decomposition. Our random also has a slight bit of pattern towards the
end. This signals that we can difference further to stabilize the
variance and remove further trends/seasonalities.

We must omit the NA values that were generated from the decomposition
function. Since we have seasonality at a period of each year, we then
find the loss at lag = 52 since we have 52 weeks in a year to see if we
can remove our seasonality component. If not, we can test our arima vs
sarima model and see if we have a lower AIC if we add in the seasonal
component to our arima.

``` r
googl_residuals <- na.omit(decompose(gtr)$random) %>% diff(lag = 52)
nvda_residuals <- na.omit(decompose(ntr)$random) %>% diff(lag = 52)
msft_residuals <- na.omit(decompose(mtr)$random) %>% diff(lag = 52)
amzn_residuals <- na.omit(decompose(atr)$random) %>% diff(lag = 52)
```

Lets check our plots to see if there is a trend/stationarity in our
residuals:

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->
It seems our plots show some trends and seasonality. We can check other
factors as well.

ACF plots:

``` r
par(mfrow = c(2,2))
acf(googl_residuals, lag.max = 100)
acf(nvda_residuals, lag.max = 100)
acf(msft_residuals, lag.max = 100)
acf(amzn_residuals, lag.max = 100)
```

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

Our acf plot suggest that we have non-stationarity since our values are
always passing the significance blue dashed line level and not gradually
decreasing to zero as lag increases. We can also check our pacf plot:

``` r
par(mfrow = c(2,2))
pacf(googl_residuals, lag.max = 100)
pacf(nvda_residuals, lag.max = 100)
pacf(msft_residuals, lag.max = 100)
pacf(amzn_residuals, lag.max = 100)
```

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

We have decaying pacf plots which is good for stationarity, but we need
to be sure we have stationarity by differencing even further because the
ACF’s indicate non-stationarity.

Let’s check our differenced plot to see if there are any trends or
seasonality:

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

Looking at our first difference, we see a significant improvement in
having a constant mean compared to our non-differenced residuals, in
addition, our plot does not have any trend or seasonality appearance

We can check our ACF plots:

``` r
par(mfrow = c(2,2))
acf(googl_residuals_diff, lag.max = 100)
acf(nvda_residuals_diff, lag.max = 100)
acf(msft_residuals_diff, lag.max = 100)
acf(amzn_residuals_diff, lag.max = 100)
```

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

We can see our acf plots are gradually converging to zero as lag
increases, suggesting some sort of stationarity processes in our
datasets, but could also suggest a hint of seasonality since there are
spikes at the periods.

Running PACF tests

``` r
par(mfrow = c(2,2))
pacf(googl_residuals_diff, lag.max = 100)
pacf(nvda_residuals_diff, lag.max = 100)
pacf(msft_residuals_diff, lag.max = 100)
pacf(amzn_residuals_diff, lag.max = 100)
```

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->
Our PACF plots aren’t decaying, and we see a small bit of seasonality.
We can check our ADF and KPSS tests to double check for stationarity,
and run our Arima model with a higher order of autoregression to account
for the pacf plot.

We run our Augmented Dickey Fuller test on each of our differenced
residuals

``` r
data.frame(GOOGL_ADF_P_VALUE = adf.test(googl_residuals_diff)$p.value,
           NVIDIA_ADF_P_VALUE = adf.test(nvda_residuals_diff)$p.value,
           MSFT_ADF_P_VALUE = adf.test(msft_residuals_diff)$p.value,
           AMZN_ADF_P_VALUE = adf.test(amzn_residuals_diff)$p.value)
```

    ##   GOOGL_ADF_P_VALUE NVIDIA_ADF_P_VALUE MSFT_ADF_P_VALUE AMZN_ADF_P_VALUE
    ## 1              0.01               0.01             0.01             0.01

The ADF test suggests that our datasets are all stationarity. We can
make sure we do not have other types of non-stationarity by running the
Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test:

``` r
data.frame(GOOGL_kpss_P_VALUE = kpss.test(googl_residuals_diff)$p.value,
           NVIDIA_kpss_P_VALUE = kpss.test(nvda_residuals_diff)$p.value,
           MSFT_kpss_P_VALUE = kpss.test(msft_residuals_diff)$p.value,
           AMZN_kpss_P_VALUE = kpss.test(amzn_residuals_diff)$p.value)
```

    ##   GOOGL_kpss_P_VALUE NVIDIA_kpss_P_VALUE MSFT_kpss_P_VALUE AMZN_kpss_P_VALUE
    ## 1                0.1                 0.1               0.1               0.1

Since our datsets pass both our ADF and KPSS tests for stationarity, we
can fit our arima time series models on our datasets.

## Fitting our Arima models:

For our model, we will use the arima model without the seasonal
component to see which models perform best. Our ranking metri will be
using aic.

GOOGLE MODELS:

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-22-1.png)<!-- -->

For our google residuals diff parameters, where we have it differenced,
we don’t need to difference it further, and so d = 0. For our seasonal
component, we can set P as 1.0 or 2.0 as they are significant at those
seasonal periods as we see from our PACF graph. For Q, we would set it
as 1 since we have a significant value at the 1.0 from the ACF graph. D
= 0 since we do not need to get the seasonal difference. For q, we see a
lot of values , where q can equal to 0, 1, 3, 10,… and p can equal 1, 3,
4, 6, 9,… and more, but to avoid overfitting our dataset and for
computational sake, we can use the first significant value where p = 1
or q = 0 or 1 , as these values also gives us the best AIC.

Therefore, here are our possibilities for our arima dataset for Google:
Sarima(p = 1 or 2, d = 0, q = 0 or 1, P = 1 or 2, D = 0, Q = 1)

After fine tuning our parameters, we ended up with the three best
models: Arima(1, 0, 1)(1, 0, 1), Arima(1, 0, 0)(1, 0, 1), and Arima(2,
0, 1)(1, 0, 1)

    ##   googl_model1.aic googl_model2.aic googl_model3.aic
    ## 1         1112.548         1114.544         1116.071

These are our best models using the arima and sarima packages. Note that
two of the parameters are the same, but are from different packages and
thus can be considered as different models.

We will now see what best fits our other datasets:

NVIDIA MODELS:

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-25-1.png)<!-- -->

For our nvidia residuals diff parameters, where we have it differenced,
we don’t need to difference it further, and so d = 0. For our seasonal
component, we can set P as 1.0 as they are significant at those seasonal
periods as we see from our PACF graph. For Q, we would set it as 1 since
we have a significant value at the 1.0 from the ACF graph. D = 0 since
we do not need to get the seasonal difference. For p, we see a lot of
values, where p can equal to 8, 9, 13,.. and q can equal 0, 8, 9,… and
more, but to avoid overfitting our dataset and for computational sake we
will consider the first three values of p and q

Therefore, here are our possibilities for our arima dataset for Nvidia:
Sarima(p = 8 or 9 d = 0, q = 0, 7, or 8 P = 1, D = 0, Q = 1)

After fine tuning our data to get the best 3 AIC values, we ended up
with the three best models: ARIMA(8, 0, 8)(1, 0, 1), AND ARIMA(9, 0,
8)(1, 0, 1), ARIMA(8, 0, 7)(1, 0, 1)

    ##   nvda_model1.aic nvda_model2.aic nvda_model3.aic
    ## 1        1069.107        1103.472        1077.038

MICROSOFT MODELS:

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-28-1.png)<!-- -->

For our msft residuals diff parameters, where we have it differenced, we
don’t need to difference it further, and so d = 0. For our seasonal
component, we can set P as 1.0 or 2.0 as they are significant at those
seasonal periods as we see from our PACF graph. For Q, we would set it
as 1 since we have a significant value at the 1.0 from the ACF graph. D
= 0 since we do not need to get the seasonal difference. For p, we see a
lot of values, where p can equal to 1, 12, 14,.. and q can equal 0, 1,
6, 17,… and more, but to avoid overfitting our dataset and for
computational sake, we will consider the first three values of p and q.

Therefore, here are our possibilities for our arima dataset for
MICROSOFT: Sarima(p = 1, 12, 14 d = 0, q = 0, 1, 6, 17 P = 1.0 or 2.0, D
= 0, Q = 1)

After fine tuning our data to get the best 3 AIC values, we ended up
with the three best models: ARIMA(12, 0, 6)(1, 0, 1), ARIMA(12, 0, 0)(1,
0, 1), ARIMA(12, 0, 1)(1, 0, 1)

    ##   msft_model1.aic msft_model2.aic msft_model3.aic
    ## 1        1340.611        1335.333        1337.569

AMAZON MODELS:

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-31-1.png)<!-- -->

For our amazn residuals diff parameters, where we have it differenced,
we don’t need to difference it further, and so d = 0. For our seasonal
component, we can set P as 1.0 or 2.0 as they are significant at those
seasonal periods as we see from our PACF graph. For Q, we would set it
as 1 since we have a significant value at the 1.0 from the ACF graph. D
= 0 since we do not need to get the seasonal difference. For p, we see a
lot of values, where p can equal to 8 and 10, 16,.. and q can equal 0,
8, 10,… and more, but to avoid overfitting our dataset and for
computational sake we will consider the first three values of p and q.

Therefore, here are our possibilities for our arima dataset for AMAZON:
Sarima(p = 8 or 9, d = 0, q = 0, 8, or 9, P = 1.0 or 2.0, D = 0, Q = 1)

After fine tuning our data to get the best 3 AIC values, we ended up
with the three best models: ARIMA(8,0,0)(1,0,1) ARIMA(8,0,0)(2,0,1)
ARIMA(9,0,0)(1,0,1)

    ##   amzn_model1.aic amzn_model2.aic amzn_model3.aic
    ## 1        1171.279        1171.458        1197.487

Now, we will test the residuals of our values to see if our models are
reliable:

## Testing models’ residuals

Our residuals should look like a normal white noise, and we can test
this using a histogram.

``` r
# Test for googl:
par(mfrow=c(2,2))
hist(googl_model1$residuals, breaks = 30)
hist(googl_model2$residuals, breaks = 30)
hist(googl_model3$residuals, breaks = 30)
```

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-34-1.png)<!-- -->

``` r
par(mfrow=c(2,2))
hist(nvda_model1$residuals, breaks = 30)
hist(nvda_model2$residuals, breaks = 30)
hist(nvda_model3$residuals, breaks = 30)
```

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-35-1.png)<!-- -->

``` r
par(mfrow = c(2,2))
hist(msft_model1$residuals, breaks = 30)
hist(msft_model2$residuals, breaks = 30)
hist(msft_model3$residuals, breaks = 30)
```

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-36-1.png)<!-- -->

``` r
par(mfrow = c(2,2))

hist(amzn_model1$residuals, breaks = 30)
hist(amzn_model2$residuals, breaks = 30)
hist(amzn_model3$residuals, breaks = 30)
```

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-37-1.png)<!-- -->

Our histograms look normal and resembles a gaussian white noise
distribution with the exception of NVIDIA, which is unaffected by our
parameters set in our arima model. We can do another test by checking
our acf and pacf graphs for all instances.

LJUNG-BOX TEST and BOX-PIERCE TESTS

    ##      Box-Pierce    Ljung-Box
    ## g1 0.0005869036 0.0003764588
    ## g2 0.0006234068 0.0004008371
    ## g3 0.0004514679 0.0002851218
    ## n1 0.9351171564 0.9173856761
    ## n2 0.0043621764 0.0025660295
    ## n3 0.8255629600 0.7858714386
    ## m1 0.9910585886 0.9880232353
    ## m2 0.9867414970 0.9816699514
    ## m3 0.9854728912 0.9800685009
    ## a1 0.4255196081 0.3743265480
    ## a2 0.4610717663 0.4092679633
    ## a3 0.1972082348 0.1601157779

These are our box pierce test values from our dataset.

PACFS AND ACF OF GOOGLE MODEL RESIDUALS:

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-40-1.png)<!-- -->

PACFS AND ACF OF NVIDIA MODEL RESIDUALS:

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-41-1.png)<!-- -->

PACFS AND ACF OF MICROSOFT MODEL RESIDUALS:

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-42-1.png)<!-- -->

PACFS AND ACF OF AMAZON MODEL RESIDUALS:

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-43-1.png)<!-- -->

All in all, our pacf and acf charts for all of our stocks do not
resemble a white noise distribution, and none of them pass our ljung box
tests, no matter how we fine tune our models, which could be a result of
our transformation that still had some trending/seasonality. This tells
us that our models may not be as effective on our test data, but we can
check by visualizing how our models perform on the test data of 2019.

## Forecasting:

#### Google Forecast:

    ## $pred
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1] 53.47938 53.37517 54.15219 55.16392 53.82021 52.74956 53.73407 54.37975
    ##  [9] 53.82974 54.81265 55.45614 53.49241 53.17454 53.07353 53.26347 53.95230
    ## [17] 53.94936 54.04051 55.24205 54.72484 55.71192 55.98895 56.33878 55.96124
    ## [25] 56.14132 55.34441 55.39723 57.78714 58.06633 58.83775 58.22608 58.24736
    ## [33] 57.76330 57.83559 58.32471 57.39663 57.59665 57.76780 57.98804 58.01464
    ## [41] 57.89426 58.18244 57.96755 58.08230 58.07986 58.07686 57.76889 58.00002
    ## [49] 58.20314 58.48827 58.26054 58.31127
    ## 
    ## $se
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1] 1.152379 1.606568 1.939969 2.208894 2.435575 2.631628 2.804082 2.957604
    ##  [9] 3.095494 3.220198 3.333593 3.437165 3.532112 3.619420 3.699916 3.774298
    ## [17] 3.843165 3.907035 3.966359 4.021535 4.072914 4.120807 4.165494 4.207225
    ## [25] 4.246225 4.282701 4.316835 4.348799 4.378745 4.406815 4.433138 4.457834
    ## [33] 4.481010 4.502770 4.523205 4.542403 4.560443 4.577398 4.593339 4.608329
    ## [41] 4.622428 4.635691 4.648170 4.659912 4.670963 4.681366 4.691158 4.700378
    ## [49] 4.709059 4.717233 4.724930 4.732202

    ## $pred
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1] 53.26561 53.16526 53.94210 54.95918 53.60382 52.54921 53.52694 54.17091
    ##  [9] 53.62919 54.60891 55.25374 53.29412 52.98187 52.88417 53.07287 53.75711
    ## [17] 53.78388 53.89040 55.07896 54.56216 55.56459 55.83579 56.18809 55.78220
    ## [25] 55.96148 55.15282 55.19494 57.59223 57.87853 58.62078 58.01455 58.02530
    ## [33] 57.54423 57.61849 58.10658 57.18409 57.37265 57.54408 57.76869 57.79956
    ## [41] 57.69704 57.98271 57.78360 57.89019 57.88669 57.88882 57.58113 57.78555
    ## [49] 58.00390 58.29417 58.06864 58.10545
    ## 
    ## $se
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1] 1.143674 1.490121 1.760142 1.985507 2.180402 2.352675 2.507229 2.647384
    ##  [9] 2.775519 2.893415 3.002446 3.103703 3.198072 3.286282 3.368947 3.446586
    ## [17] 3.519646 3.588514 3.653528 3.714987 3.773155 3.828268 3.880539 3.930159
    ## [25] 3.977300 4.022120 4.064762 4.105359 4.144030 4.180887 4.216033 4.249562
    ## [33] 4.281563 4.312118 4.341303 4.369189 4.395844 4.421328 4.445701 4.469018
    ## [41] 4.491330 4.512685 4.533129 4.552705 4.571454 4.589414 4.606621 4.623110
    ## [49] 4.638914 4.654062 4.668580 4.682518

    ## $pred
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1] 53.23557 53.42090 53.95436 55.26098 53.62044 52.72995 53.53440 54.40577
    ##  [9] 53.61898 54.86247 55.32726 53.48755 52.99275 53.06789 53.12010 53.98407
    ## [17] 53.85879 54.10050 55.20287 54.79747 55.69598 56.11017 56.36553 56.08459
    ## [25] 56.16834 55.43077 55.37485 57.89556 58.09272 58.97724 58.26244 58.36338
    ## [33] 57.78242 57.94368 58.38141 57.47409 57.60102 57.84739 58.02600 58.09689
    ## [41] 57.90427 58.22434 57.94041 58.13716 58.08393 58.12477 57.75041 58.03736
    ## [49] 58.20746 58.53073 58.24569 58.33736
    ## 
    ## $se
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1] 1.149876 1.581204 1.918938 2.176454 2.407038 2.595765 2.771808 2.920305
    ##  [9] 3.061539 3.182723 3.299256 3.400368 3.498269 3.583895 3.667175 3.740460
    ## [17] 3.811952 3.875169 3.936967 3.991832 4.045540 4.093384 4.140263 4.182145
    ## [25] 4.223207 4.259986 4.296055 4.328437 4.360197 4.388769 4.416791 4.442048
    ## [33] 4.466814 4.489175 4.511096 4.530919 4.550345 4.567938 4.585173 4.600802
    ## [41] 4.616106 4.630003 4.643603 4.655968 4.668062 4.679071 4.689833 4.699639
    ## [49] 4.709220 4.717960 4.726493 4.734296

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-44-1.png)<!-- -->

#### Nvidia Forecast:

    ## $pred
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1] 34.00326 34.87516 35.52663 36.71819 36.92392 37.30477 37.48586 36.97857
    ##  [9] 35.46850 36.99886 38.02906 37.51497 36.22378 34.74699 36.48814 35.94003
    ## [17] 36.24231 36.31519 39.74366 40.17781 40.18123 41.11585 42.79477 42.67621
    ## [25] 42.27218 40.28134 40.11661 42.07556 42.35994 42.59027 42.64312 42.44459
    ## [33] 42.75681 44.31280 45.54405 44.55376 45.55648 45.22060 45.85839 46.49429
    ## [41] 43.98671 43.97835 39.80596 42.18052 41.39698 38.08892 36.31603 36.65026
    ## [49] 36.71339 35.52816 35.01489 35.00812
    ## 
    ## $se
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1] 1.483838 2.138545 2.629699 3.037760 3.392895 3.710382 3.999242 4.265336
    ##  [9] 4.512732 4.744392 4.962554 5.168960 5.364998 5.551797 5.730286 5.901248
    ## [17] 6.065346 6.223146 6.375141 6.521760 6.663379 6.800334 6.932922 7.061409
    ## [25] 7.186034 7.307016 7.424549 7.538814 7.649976 7.758183 7.863577 7.966283
    ## [33] 8.066421 8.164101 8.259426 8.352490 8.443383 8.532188 8.618985 8.703846
    ## [41] 8.786840 8.868034 8.947489 9.025262 9.101411 9.175985 9.249037 9.320611
    ## [49] 9.390755 9.459510 9.526919 9.593033

    ## $pred
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1] 34.10300 34.98563 35.63731 36.83514 37.03031 37.40665 37.60955 37.11043
    ##  [9] 35.57461 37.12165 38.15189 37.62035 36.30922 34.82649 36.59706 36.03058
    ## [17] 36.32596 36.40370 39.82356 40.23671 40.23391 41.16501 42.82326 42.72416
    ## [25] 42.30303 40.30719 40.13925 42.08775 42.36262 42.60402 42.63939 42.45224
    ## [33] 42.75449 44.32410 45.56099 44.56548 45.56498 45.20606 45.85020 46.49110
    ## [41] 43.92304 43.90156 39.67206 42.05954 41.24091 37.87388 36.07174 36.44875
    ## [49] 36.51383 35.31803 34.77817 34.75785
    ## 
    ## $se
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1] 1.485474 2.094639 2.557916 2.945029 3.283077 3.586002 3.862116 4.116851
    ##  [9] 4.353993 4.576308 4.785892 4.984378 5.173069 5.353025 5.525125 5.690103
    ## [17] 5.848584 6.001103 6.148126 6.290055 6.427249 6.560022 6.688656 6.813401
    ## [25] 6.934485 7.052113 7.166470 7.277726 7.386036 7.491543 7.594377 7.694659
    ## [33] 7.792503 7.888011 7.981280 8.072401 8.161457 8.248529 8.333689 8.417007
    ## [41] 8.498550 8.578378 8.656550 8.733120 8.808142 8.881665 8.953734 9.024396
    ## [49] 9.093693 9.161664 9.228348 9.293800

    ## $pred
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1] 33.97041 34.82825 35.48018 36.67198 36.87857 37.26002 37.44060 36.93332
    ##  [9] 35.42458 36.95475 37.98540 37.47236 36.18227 34.70595 36.44638 35.89933
    ## [17] 36.20228 36.27531 39.70476 40.14014 40.14414 41.07931 42.75956 42.64049
    ## [25] 42.23747 40.24698 40.08266 42.04251 42.32761 42.55779 42.61164 42.41289
    ## [33] 42.72583 44.28167 45.51306 44.52318 45.52638 45.19169 45.82953 46.46552
    ## [41] 43.96051 43.95292 39.78280 42.15726 41.37533 38.06963 36.29804 36.63077
    ## [49] 36.69405 35.50930 34.99724 34.99122
    ## 
    ## $se
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1] 1.483744 2.138307 2.637149 3.050636 3.410018 3.731013 4.022869 4.291576
    ##  [9] 4.541288 4.775023 4.995060 5.203171 5.400768 5.588996 5.768802 5.940980
    ## [17] 6.106201 6.265043 6.418003 6.565518 6.707970 6.845698 6.979004 7.108158
    ## [25] 7.233402 7.354957 7.473022 7.587779 7.699395 7.808021 7.913800 8.016860
    ## [33] 8.117321 8.215296 8.310888 8.404194 8.495304 8.584302 8.671269 8.756279
    ## [41] 8.839402 8.920705 9.000249 9.078095 9.154297 9.228910 9.301983 9.373564
    ## [49] 9.443700 9.512433 9.579805 9.645872

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-45-1.png)<!-- -->

#### Microsoft Forecast:

    ## Warning in stats::arima(xdata, order = c(p, d, q), seasonal = list(order = c(P,
    ## : possible convergence problem: optim gave code = 1

    ## $pred
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1] 102.6548 102.5980 102.9118 105.2996 102.7551 101.5708 103.5140 103.2180
    ##  [9] 103.3653 103.2570 104.3697 102.8611 102.4661 103.7534 103.4221 104.1997
    ## [17] 105.9772 103.6197 106.2591 104.3708 105.6412 106.8448 106.4328 106.7678
    ## [25] 106.0657 104.7533 106.0704 107.8679 110.0677 110.6298 109.2705 110.5557
    ## [33] 109.0781 109.4407 111.2656 109.3549 112.0509 111.2973 111.1502 112.1490
    ## [41] 110.3755 112.7531 113.2252 112.9252 114.6279 112.8346 111.8668 114.3449
    ## [49] 113.2739 114.9486 112.3076 112.5077
    ## 
    ## $se
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1] 1.630120 2.107820 2.486350 2.837434 3.150595 3.385100 3.646234 3.882387
    ##  [9] 4.130747 4.384554 4.607601 4.823248 5.001517 5.181451 5.370785 5.550614
    ## [17] 5.735946 5.895059 6.040606 6.191538 6.334698 6.491167 6.636434 6.765905
    ## [25] 6.895555 7.013587 7.142639 7.272768 7.391341 7.510118 7.614105 7.721521
    ## [33] 7.834834 7.941585 8.052421 8.149968 8.243607 8.342176 8.435822 8.536934
    ## [41] 8.630580 8.716650 8.805192 8.887301 8.976605 9.065002 9.146113 9.228866
    ## [49] 9.303264 9.381260 9.462256 9.537984

    ## $pred
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1] 101.5054 101.4707 102.3680 104.5467 102.8286 100.4338 102.5441 102.8591
    ##  [9] 101.7934 102.8182 103.2849 101.4267 101.7903 102.0151 102.7619 103.3232
    ## [17] 104.4658 102.9861 104.5405 103.3220 104.7982 105.2228 105.8541 105.2434
    ## [25] 104.7668 103.9812 104.4024 107.2468 108.7750 109.2046 108.5322 108.8588
    ## [33] 108.2903 108.3554 109.7293 108.6437 110.3847 110.3203 110.2845 110.5476
    ## [41] 109.6268 111.0693 111.7040 111.8418 112.7968 111.8906 110.1948 112.7366
    ## [49] 112.2382 112.9819 111.1075 110.9721
    ## 
    ## $se
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1] 1.663114 2.141734 2.526280 2.876792 3.166694 3.400305 3.660240 3.910274
    ##  [9] 4.168695 4.404022 4.629320 4.857016 5.065570 5.264074 5.457658 5.641983
    ## [17] 5.818826 5.990906 6.156771 6.317514 6.472450 6.622530 6.768247 6.909271
    ## [25] 7.046217 7.179495 7.308990 7.435023 7.557819 7.677458 7.794126 7.907937
    ## [33] 8.019041 8.127562 8.233588 8.337241 8.438621 8.537802 8.634878 8.729930
    ## [41] 8.823027 8.914241 9.003638 9.091282 9.177229 9.261536 9.344252 9.425433
    ## [49] 9.505123 9.583368 9.660207 9.735695

    ## $pred
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1] 101.4794 101.4594 102.5017 104.5738 102.6846 100.5006 102.7297 102.6837
    ##  [9] 101.8753 102.6801 103.3980 101.3687 101.8680 102.0226 102.8041 103.2824
    ## [17] 104.4999 102.9840 104.5276 103.3316 104.7775 105.2125 105.8110 105.2264
    ## [25] 104.7424 103.9764 104.3858 107.2167 108.7065 109.1364 108.4715 108.7971
    ## [33] 108.2386 108.2980 109.6449 108.5855 110.3006 110.2342 110.2019 110.4681
    ## [41] 109.5546 111.0074 111.6363 111.7783 112.7112 111.8207 110.1498 112.6556
    ## [49] 112.1640 112.9034 111.0596 110.9209
    ## 
    ## $se
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1] 1.659443 2.139232 2.523027 2.872777 3.161786 3.394450 3.655806 3.905833
    ##  [9] 4.163302 4.398727 4.625765 4.862186 5.063084 5.265958 5.457852 5.642381
    ## [17] 5.818683 5.992649 6.157878 6.319694 6.474169 6.624987 6.770963 6.911695
    ## [25] 7.049355 7.182654 7.312433 7.438692 7.561873 7.681621 7.798676 7.912630
    ## [33] 8.024095 8.132827 8.239135 8.343084 8.444731 8.544195 8.641563 8.736904
    ## [41] 8.830285 8.921806 9.011488 9.099443 9.185683 9.270300 9.353321 9.434811
    ## [49] 9.514812 9.593370 9.670522 9.746329

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-46-1.png)<!-- -->

#### Amazon Forecast:

    ## $pred
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1] 75.20012 75.41663 75.78935 76.09745 76.24745 74.01561 75.49481 76.38496
    ##  [9] 75.85374 77.02177 76.92799 76.14028 74.46018 74.89334 75.08870 77.22944
    ## [17] 79.21360 77.71169 79.05457 78.51039 79.35412 79.98349 80.75455 80.91559
    ## [25] 81.15560 80.51326 80.13052 82.53478 83.44538 83.97127 82.84576 83.15064
    ## [33] 82.89052 83.16151 85.10011 83.97747 84.98399 84.05166 84.85921 83.82398
    ## [41] 82.54323 81.87719 80.53975 81.35038 82.54776 80.65725 80.12613 82.31290
    ## [49] 82.72806 81.76484 78.84149 79.05355
    ## 
    ## $se
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1]  1.511703  2.135902  2.544135  2.902332  3.269234  3.694433  4.071228
    ##  [8]  4.454978  4.807687  5.135525  5.451087  5.750721  6.041292  6.314194
    ## [15]  6.574227  6.821735  7.057366  7.282901  7.497883  7.703669  7.900259
    ## [22]  8.088462  8.268870  8.441831  8.607901  8.767354  8.920639  9.068072
    ## [29]  9.209976  9.346661  9.478381  9.605404  9.727954  9.846257  9.960517
    ## [36] 10.070923 10.177654 10.280874 10.380742 10.477402 10.570991 10.661639
    ## [43] 10.749466 10.834588 10.917112 10.997140 11.074769 11.150090 11.223188
    ## [50] 11.294147 11.363044 11.429956

    ## $pred
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1]  76.36389  79.22067  80.29135  83.57804  86.54480  81.52916  85.52739
    ##  [8]  86.86213  85.20352  89.73194  90.34570  88.03053  82.80603  83.69383
    ## [15]  83.99618  89.10443  93.24872  89.71718  91.83856  90.60230  92.14107
    ## [22]  93.59893  95.85788  96.91224  98.45584  96.49584  95.03365 100.44835
    ## [29] 101.42920 103.55610 101.28162 102.84110 102.79510 103.65976 108.63097
    ## [36] 105.17138 107.89119 105.11688 107.28785 103.44157  98.94569  97.52056
    ## [43]  92.71378  94.80965  98.11099  91.38460  88.32644  96.42631  97.46065
    ## [50]  94.46038  86.27018  86.84764
    ## 
    ## $se
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1]  1.781239  2.537085  3.060347  3.515287  3.966303  4.447681  4.874473
    ##  [8]  5.301467  5.692977  6.054147  6.393263  6.710946  7.012487  7.293246
    ## [15]  7.557366  7.805299  8.038318  8.257936  8.464751  8.660030  8.844274
    ## [22]  9.018398  9.183101  9.339005  9.486733  9.626785  9.759681  9.885860
    ## [29] 10.005744 10.119719 10.228137 10.331327 10.429589 10.523205 10.612435
    ## [36] 10.697521 10.778686 10.856142 10.930083 11.000695 11.068146 11.132598
    ## [43] 11.194200 11.253094 11.309427 11.363299 11.414866 11.464212 11.511436
    ## [50] 11.556629 11.599910 11.641516

    ## $pred
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1] 75.41590 76.81819 76.89253 75.60861 76.51399 75.50135 77.62838 77.59186
    ##  [9] 76.96842 78.39002 78.62027 78.36573 76.63748 77.18112 77.45898 79.74089
    ## [17] 81.75692 80.36967 81.79496 81.38512 82.42260 83.07651 83.86584 84.06005
    ## [25] 84.35058 83.90341 83.64775 85.99473 86.98436 87.51823 86.55268 86.85106
    ## [33] 86.68529 87.00431 88.87728 87.97950 88.98652 88.21762 89.02222 88.25219
    ## [41] 87.27968 86.74400 85.76282 86.60982 87.79961 86.28504 85.99418 87.93237
    ## [49] 88.37930 87.62724 85.10246 85.37888
    ## 
    ## $se
    ## Time Series:
    ## Start = c(2019, 3) 
    ## End = c(2020, 2) 
    ## Frequency = 52 
    ##  [1]  1.499110  2.137935  2.551417  2.917779  3.290698  3.732943  4.129577
    ##  [8]  4.523588  4.832385  5.173574  5.519433  5.840567  6.147885  6.432484
    ## [15]  6.720476  6.995494  7.267908  7.524716  7.773155  8.018754  8.256356
    ## [22]  8.489544  8.713951  8.934437  9.149402  9.360383  9.566840  9.768124
    ## [29]  9.966132 10.159972 10.350763 10.537761 10.721582 10.902341 11.080138
    ## [36] 11.255332 11.427643 11.597519 11.764865 11.929944 12.092767 12.253399
    ## [43] 12.411985 12.568532 12.723205 12.875990 13.026998 13.176267 13.323864
    ## [50] 13.469855 13.614270 13.757179

![](Time-Series-Analysis_files/figure-gfm/unnamed-chunk-47-1.png)<!-- -->

## Forecast Results, Findings, and Inferences:

Our Google stock had a smooth line, so our prediction assumed that we
did not have a seasonality and instead a gradual, positive trend. The
arima model prediction was fairly close with Google’s true value in blue
until the 3rd quarter of 2019, when Google’s stock increased much more
than predicted. The same applies for all of our arima models fitted on
the google dataset.

Our Nvidia stock forecast did manage to imitate the true values.
However, it veered off trajectory in the 3rd quarter of 2019.
Nevertheless, this is our second best prediction and it performed fairly
well.

Our Microsoft stock prediction were far off from what we had predicted
despite its stock having a steady increase. The forecast most likely
used its current stagnant trend to make a prediction, which was when the
stock decided to increase in price since the beginning of 2019.

Our Amazon stock forecast was our most accurate forecast. Albeit having
a stable trend, the prediction accurately determined its trajectory up
until the end of 2020. Our 2nd Arima model correctly guessed the
majority of Amazon’s price trajectory throughout 2019. We can see that
the true value was within our forecast’s confidence bands for almost the
entirety of the true values, though this is not much as these stocks are
not very volatile.

Overall, we can conclude that we did not achieve the forecasts that we
wanted, but if we were to forecast our stocks, we would run our arima
models on the Amazon stock for its predictability. It seemed our
forecasts did also did not account for the fact that we had seasonality
in our dataset. It seems the higher the values of our p, q, P, and Q,
the lesser the accurate we are in our models. This could be due to the
fact that we have overfit on our training set and thus performed poorly
on the test set. We also have shown that we need to find a problem with
either our transformation or our model, because we did not get gaussian
white noise distributions from our residuals after we fit the data. All
in all, the forecasts and models did not fit, but some made decent
predictions, especially Amazon. We can also infer from the fact that
these are incredibly stable stocks and not as volatile as other riskier
stocks. Thus, we have a more stable true prediction which could
contribute to having a good forecast despite our evaluation of our
models not being very strong.

There were other noticeable factors that could have played into our poor
stock forecast as well. One was notably the US-China trade war, which
began in mid-July which was towards the end of our training dataset.
This event crippled the technology industry especially, with the
industry being heavily reliant on labor and manufacturing parts from
China.

## Conclusion:

In conclusion, our analysis of the weekly stock prices for four of the
largest technology companies in the world over a 10-year period from
2010 to 2020 has provided us with interesting insights into the
performance of these stocks. Our main objective was to determine which
of the four stocks would have the highest predicted performance in the
year 2019, and thus we used ARIMA models to make predictions.

After analyzing the data and comparing our predictions to the true
values, we found that our models did not perform as accurately as we had
hoped. Our Google stock had a smooth line, and our prediction assumed
that we did not have seasonality, but our ARIMA model was fairly close
until the 3rd quarter of 2019, when Google’s stock increased more than
predicted. Our Nvidia stock forecast managed to imitate the true values
until it veered off course towards the 3rd quarter of 2019 as well, but
our Microsoft stock prediction was far off from what we had predicted
despite its stock having a steady increase. Finally, our Amazon stock
forecast was our most accurate forecast, with the second ARIMA model
making correct predictions of the stock’s price trajectory throughout
2019.

We can conclude that our ARIMA models did not perform as accurately as
we had hoped, but our analysis did provide us with some valuable
insights. For instance, we observed that Amazon was the most predictable
stock among the four, and thus it could be an excellent choice for
investors who wish to invest in individual stocks from the technology
industry.

Overall, our analysis highlighted the importance of using statistical
models to make stock predictions, especially in industries as dynamic as
the technology sector. While our models did not provide us with the
accuracy we were hoping for, it is still a valuable exercise in
understanding the nuances of the stock market and making informed
investment decisions. Future analyses could include more factors and
consider the impact of major events on the stocks, such as pandemics,
economic crises, and geopolitical tensions.

## Appendix:

Datasets retrieved from Yahoo Finance:

Amazon. (n.d.). Retrieved from
<https://finance.yahoo.com/quote/AMZN/history?p=AMZN>

Microsoft. (n.d.). Retrieved from
<https://finance.yahoo.com/quote/MSFT/history?p=MSFT>

Nvidia. (n.d.). Retrieved from
<https://finance.yahoo.com/quote/NVDA/history?p=NVDA>

Google. (n.d.). Retrieved from
<https://finance.yahoo.com/quote/GOOGL/history?p=GOOGL>
