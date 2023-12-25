import { Component, OnInit } from '@angular/core';

//TensorFlow.js
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

// import { TfjsMetricsService } from 'projects/tfjs-metrics/src/public-api';
import { TfjsMetricsService } from 'tfjs-metrics';

const csvUrl =
  'https://docs.google.com/spreadsheets/d/e/2PACX-1vSUgRO2FRhFUA6ycWjKiol5mqfHPWcPuwOmJJxPbMT4PLOa86Bj_dobndkogPRWrTce8VeKDIVjXr6B/pub?gid=1111655281&single=true&output=csv';


@Component({
  selector: 'app-model7',
  templateUrl: './model7.component.html',
  styleUrls: ['./model7.component.scss']
})
export class Model7Component implements OnInit {

  dataset!: any;
 
 


  constructor(private readonly tfmetrics: TfjsMetricsService){}

  ngOnInit(): void {
    this.loadData();
    this.visualizeDataset();
    // this.defineAndTrainModel();
  }

  loadData() {
    // Our target variable (what we want to predict) is the column 'label' (wow, very original),
    // so we specify it in the configuration object as the label
    this.dataset = tf.data.csv(csvUrl, {
      columnConfigs: {
        complication: {
          isLabel: true,
        },
      },
    });
  }

  features_array: any = [];
  target_array: any = [];

  features_test: any = [];
  target_test: any = [];


  async visualizeDataset() {
    // tfjs-vis surface's names and tabs
    const dataSurface = { name: 'Dataset Surgical binary classification (Kaggle)', tab: 'Charts' };
    const classZero: any = [];
    const classOne: any = [];

    let counter_class_0 = 0, counter_class_1 = 0;
    const number_of_samples = 200;    
   
    let test_counter_complication =0;
    let test_counter_non_complication =0;


    await this.dataset.forEachAsync((e: any) => {
      // Extract the features from the dataset
      const features = { x: e.xs.bmi, y: e.xs.gender };

      // If the label is 0, add the features to the "classZero" array
      if ((e.ys.complication === 0) && (counter_class_0 < number_of_samples) && (Math.random() < 0.5)) {
       
        counter_class_0++;
        classZero.push(features);

        //Adding to global variable for training
        this.features_array.push(features);
        this.target_array.push(e.ys.complication);

      } else if ((e.ys.complication === 1) && (counter_class_1 < number_of_samples) && (Math.random() < 0.5)) {

        counter_class_1++;
        classOne.push(features);

        //Adding to global variable for training
        this.features_array.push(features);
        this.target_array.push(e.ys.complication);

      } else if ((e.ys.complication === 1) && (test_counter_complication < 50) && (Math.random() < 0.5)){

        this.features_test.push(Object.values(e.xs));
        this.target_test.push(e.ys.complication);
        test_counter_complication++;

      } else if ((e.ys.complication === 0) && (test_counter_non_complication < 50) && (Math.random() < 0.5)){

        this.features_test.push(Object.values(e.xs));
        this.target_test.push(e.ys.complication);
        test_counter_non_complication++;
        

      }
    });

    // console.log("Target for testing: ",  target_test);

    // Specify the name of the labels. This will show up in the chart's legend.
    const series = ['No Complication', 'Complication'];
    const dataToDraw = { values: [classZero, classOne], series };

    tfvis.render.scatterplot(dataSurface, dataToDraw, {
      xLabel: 'Body Mass Index',
      yLabel: 'Age',
      zoomToFit: true,
    });

    this.defineAndTrainModel();
  }//visualizeDataset()

  async defineAndTrainModel() {

    /**Pre-preparation */
    const numberEpochs = 100;
    // numOfFeatures is the number of column or features minus the label column
    const numOfFeatures = (await this.dataset.columnNames()).length - 1;
    const trainingSurface = { name: 'Loss and MSE', tab: 'Training' };

    //Creating tensors
    await this.dataset_to_array();

    this.shuffle(this.features_array, this.target_array);

    const features_tensor_raw = tf.tensor2d(this.features_array, [this.features_array.length, numOfFeatures]);
    const target_tensor = tf.tensor2d(this.target_array, [this.target_array.length, 1]);


    let { dataMean, dataStd } = this.determineMeanAndStddev(features_tensor_raw);
    const features_tensor = this.normalizeTensor(features_tensor_raw, dataMean, dataStd);


    //Defining the model
    const model = tf.sequential();

    //Adding layers
    //Input layer
    model.add(
      tf.layers.dense({
        inputShape: [numOfFeatures],
        units: 200,
        activation: 'relu',
      })
    );

    // model.add(
    //   tf.layers.dense({
    //     inputShape: [numOfFeatures],
    //     units: 100,
    //     activation: 'relu',
    //   })
    // );

    //Output Layer
    model.add(
      tf.layers.dense(
        { units: 1, activation: 'sigmoid' }
      ));


    model.compile({
      optimizer: tf.train.adam(),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy'],
    });

    // Print the summary to console
    model.summary();

    // Fit the model
    await model.fit(features_tensor_raw, target_tensor, {
      batchSize: 40,
      epochs: numberEpochs,
      validationSplit: 0.2,

      callbacks: [
        // Show on a tfjs-vis visor the loss and accuracy values at the end of each epoch.
        tfvis.show.fitCallbacks(trainingSurface, ['loss', 'acc', "val_loss", "val_acc"], {
          callbacks: ['onEpochEnd'],
        }),
        {
          // Print to console the loss value at the end of each epoch.
          onEpochEnd: async (epoch: any, logs: any) => {
            console.log(`${epoch}:${logs.loss}`);
          },
        },
        {
          onTrainEnd: async () => {
            console.log('Training has ended.');
          },
        },
      ],
    });

    // // Output value should be near 0.
    // (model.predict(tf.tensor2d([[...this.class_0_sample]])) as tf.Tensor).print();
    // // Output value should be near 1.
    // (model.predict(tf.tensor2d([[...this.class_1_sample]])) as tf.Tensor).print();

    
    console.log("Samples for testing: ", this.features_test);
    
    //Confusion matrix
    const confusion_matrix: any =  await this.tfmetrics.confusionMatrix(model, this.features_test, this.target_test);
    console.log([confusion_matrix]);
    //Performance metrics
    
    const performance= this.tfmetrics.performance_metrics(confusion_matrix);
    console.log("Performance: ", performance);

    //Downloading the final model    
    await model.save('downloads://my-model');

  }

  class_0_sample: any;
  class_1_sample: any;


  async dataset_to_array() {
    const features: any = [];
    const target: any = [];

    let counter_class_0 = 0;
    let counter_class_1 = 0;

    /*** */
    const number_of_samples = 500;

    await this.dataset.forEachAsync((e: any) => {
      if ((e.ys.complication === 0) && (counter_class_0 < number_of_samples) && (Math.random() < 0.5)) {

        if ((!this.class_0_sample) && (Math.random() < 0.5))
          this.class_0_sample = Object.values(e.xs);

        features.push(Object.values(e.xs));
        target.push(e.ys.complication);
        counter_class_0++;
      }
      else if ((e.ys.complication === 1) && (counter_class_1 < number_of_samples) && (Math.random() < 0.5)) {


        if ((!this.class_1_sample) && (Math.random() < 0.5))
          this.class_1_sample = Object.values(e.xs);

        counter_class_1++;
        features.push(Object.values(e.xs));
        target.push(e.ys.complication);
      }

    });

    this.features_array = features;
    this.target_array = target;
  }

  shuffle(data: any, target: any) {
    let counter = data.length;
    let temp = 0;
    let index = 0;
    while (counter > 0) {
      index = (Math.random() * counter) | 0;
      counter--;
      // data:
      temp = data[counter];
      data[counter] = data[index];
      data[index] = temp;
      // target:
      temp = target[counter];
      target[counter] = target[index];
      target[index] = temp;
    }
  };

  normalizeTensor(data: tf.Tensor, dataMean: tf.Tensor, dataStd: tf.Tensor) {
    return data.sub(dataMean).div(dataStd);
  }

  determineMeanAndStddev(data: tf.Tensor) {
    const dataMean = data.mean(0);
    // TODO(bileschi): Simplify when and if tf.var / tf.std added to the API.
    const diffFromMean = data.sub(dataMean);
    const squaredDiffFromMean = diffFromMean.square();
    const variance = squaredDiffFromMean.mean(0);
    const dataStd = variance.sqrt();
    return { dataMean, dataStd };
  }

}//end of Model7Component
