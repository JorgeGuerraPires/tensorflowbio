# TfjsMetrics

This package was created to support people working with TensorFlow.js. Different from TensorFlow in Python, we do not have such a library. When you publish a scientific paper, you may be asked to calculate those measures. Even though you can use `@tensorflow/tfjs-vis` to see the loss function and the accuracy, reviewers may ask for more.

This package will give you already ready to use a set of basic metrics for scientific papers.


## Getting ready

It is a service where we implemented the metrics.

Import the service to your component
`import { TfjsMetricsService } from 'tfjs-metrics';`


Then, inject the service on your component:

`  constructor(private readonly tfmetrics: TfjsMetricsService){}`

Now, it is ready to be called, and used!

## Confusion matrix 

This matrix will calculate how your model is doing, if it is confusing the classes.

`const confusion_matrix: any =  await this.tfmetrics.confusionMatrix(model, features_test, target_test)` 

Where: `model` is your TFJS model, trained. `features_test` are your features, `features_test` are the respective labels.

Tip. separate those samples from the training process, for making sure your model was able to generalize. Those are vectors and matrix, not tensors. We transform into tensors ourselves. It was tested for numbers, not yet for computer vision.

Tip. the features and target should be arrays. 
E.g.,
With two features
feature = [[1 2 3], [2 3 4]]
target = [ 1 0]

Real example:

This example is when you are using `await this.dataset.forEachAsync`, an internal routine from TFJS for working with datasets uploaded from CSV files. 


` features_test.push(Object.values(e.xs));`

There is an array to store the features; and also the target/labels. `Object.values`   will transform the object into array elements, for pushing, in case it is as an object. 

Then, we also save the respective label, which we need to compare the expected label vs. predicted one by the model.

` target_test.push(e.ys.diabetes);`


See paper: [Machine learning in medicine using JavaScript: building web apps using TensorFlow.js for interpreting biomedical datasets] (https://www.medrxiv.org/content/10.1101/2023.06.21.23291717v2.full)

## Basic training metrics

We should measure how our models generalized. Those are basic metrics you can use, and they are largely accepted.

`const performance= this.tfmetrics.performance_metrics(confusion_matrix)`

Where: `confusion_matrix` is the confusion matrix. 

It will calculate: 

- F1 score;
- recall
- precision


## Limitations

Those metrics were created for binary classification, it will not work as intented for multi-class classification problems.


## Further help

Let me know your thoughts on jorgeguerrabrazil@gmail.com