# Distracted-Driver-Detection

Problem Description
In this competition you are given driver images, each taken in a car with a driver doing something in the car (texting, eating, talking on the phone, makeup, reaching behind, etc). Your goal is to predict the likelihood of what the driver is doing in each picture.

The 10 classes to predict are as follows,

c0: safe driving

c1: texting - right

c2: talking on the phone - right

c3: texting - left

c4: talking on the phone - left

c5: operating the radio

c6: drinking

c7: reaching behind

c8: hair and makeup

c9: talking to passenger


Summary of Results
Using a 50-layer Residual Network (with the following parameters) the following scores (losses) were obtained.

10 Epochs
32 Batch Size
Adam Optimizer
Glorot Uniform Initializer
**Training Loss**	0.93
**Validation Loss**	3.79
**Holdout Loss**	2.64
Why the high losses? Simply put - we don't have enough resources to quickly iterate / hyper-parameter tune the model! If more resources were available (RAM, CPU speed), we could hyper-parameter tune over grid searches and combat high bias / high variance, which this model currently suffers. This is how you'd fix high bias/variance.

