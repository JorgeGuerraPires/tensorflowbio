import { Component, OnInit } from '@angular/core';

//TensorFlow.js
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';


const csvUrl =
  'https://docs.google.com/spreadsheets/d/e/2PACX-1vRyCGmhrKkYT4v6s52NWG5tU0_Y54coitcv-tAah1SNm0Pq2jhPkSUOA6vg29cHV8aqgZH_KnbO5gGp/pub?gid=1371905326&single=true&output=csv';


@Component({
  selector: 'app-model8',
  templateUrl: './model8.component.html',
  styleUrls: ['./model8.component.scss']
})
export class Model8Component implements OnInit {

  dataset!: any;


  ngOnInit(): void {
    this.loadData();
    this.visualizeDataset();
  }

  loadData() {
    // Our target variable (what we want to predict) is the column 'label' (wow, very original),
    // so we specify it in the configuration object as the label
    this.dataset = tf.data.csv(csvUrl, {
      columnConfigs: {
        HeartDisease: {
          isLabel: true,
        },
      },
    });
  }

  async visualizeDataset() {
    // tfjs-vis surface's names and tabs
    const dataSurface = { name: 'Heart Failure Prediction Dataset (Kaggle)', tab: 'Charts' };
    const classZero: any = [];
    const classOne: any = [];

    let counter_class_0 = 0, counter_class_1 = 0;
    const number_of_samples = 400;

    await this.dataset.forEachAsync((e: any) => {

      //Changing letters to numbers to sex
      if (e.xs.Sex === "M")
        e.xs.Sex = 1;
      else
        e.xs.Sex = 0;

      //Changing letters to Exercise Angina
      if (e.xs.ExerciseAngina === "Y")
        e.xs.ExerciseAngina = 1;
      else
        e.xs.ExerciseAngina = 0;

      // Extract the features from the dataset
      const features = { x: e.xs.Age, y: e.xs.ExerciseAngina };

      // If the label is 0, add the features to the "classZero" array
      if ((e.ys.HeartDisease === 0) && (counter_class_0 < number_of_samples)) {
        counter_class_0++;
        classZero.push(features);

      } else if ((e.ys.HeartDisease === 1) && (counter_class_1 < number_of_samples)) {

        counter_class_1++;
        classOne.push(features);
      }
    });

    // Specify the name of the labels. This will show up in the chart's legend.
    const series = ['Normal', 'coronary heart disease (risk)'];
    const dataToDraw = { values: [classZero, classOne], series };

    tfvis.render.scatterplot(dataSurface, dataToDraw, {
      xLabel: 'Age',
      yLabel: 'Cholesterol Level',
      zoomToFit: true,
    });

    console.log("Class 0", counter_class_0);
    console.log("Class 1 ", counter_class_1);
    this.defineAndTrainModel();
  }//visualizeDataset()

  async defineAndTrainModel() {

    /**Pre-preparation */
    const numberEpochs = 200;
    // numOfFeatures is the number of column or features minus the label column
    const numOfFeatures = (await this.dataset.columnNames()).length - 1;
    const trainingSurface = { name: 'Loss and MSE', tab: 'Training' };

    //Creating tensors
    await this.dataset_to_array();

    //Shuffle the dataset
    this.shuffle(this.features_array, this.target_array);

    const features_tensor_raw = tf.tensor2d(this.features_array, [this.features_array.length, numOfFeatures]);
    const target_tensor = tf.tensor2d(this.target_array, [this.target_array.length, 1]);


    let { dataMean, dataStd } = this.determineMeanAndStddev(features_tensor_raw);
    const features_tensor_normalized = this.normalizeTensor(features_tensor_raw, dataMean, dataStd);


    //Defining the model
    const model = tf.sequential();

    //Adding layers
    //Input layer
    model.add(
      tf.layers.dense({
        inputShape: [numOfFeatures],
        units: 100,
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
            // console.log(`${epoch}:${logs.loss}`);
            console.log(`${epoch}:${logs}`);
          },
        },
        {
          onTrainEnd: async () => {
            console.log('Training has ended.');
          },
        },
      ],
    });


    // Output value should be near 0.
    (model.predict(tf.tensor2d([[...this.class_0_sample]])) as tf.Tensor).print();
    // Output value should be near 1.
    (model.predict(tf.tensor2d([[...this.class_1_sample]])) as tf.Tensor).print();

    // (model.predict(this.normalizeTensor(tf.tensor2d([[...this.class_0_sample]]), dataMean, dataStd)) as tf.Tensor).print();
    // // Output value should be near 1.
    // (model.predict(this.normalizeTensor(tf.tensor2d([[...this.class_1_sample]]), dataMean, dataStd)) as tf.Tensor).print();
  }

  features_array: any = [];
  target_array: any = [];

  class_0_sample: any;
  class_1_sample: any;

  /**This method will transform the input dataset into an matrix with data */
  async dataset_to_array() {
    const features: any = [];
    const target: any = [];

    let counter_class_0 = 0;
    let counter_class_1 = 0;

    /*** */
    const number_of_samples = 500;

    await this.dataset.forEachAsync((e: any) => {


      //Changing letters to Exercise Angina
      if (e.xs.ExerciseAngina === "Y")
        e.xs.ExerciseAngina = 1;
      else
        e.xs.ExerciseAngina = 0;

      //Changing letters to Sex
      if (e.xs.Sex === "M")
        e.xs.Sex = 1;
      else
        e.xs.Sex = 0;

      if ((e.ys.HeartDisease === 0) && (counter_class_0 < number_of_samples) && (Math.random() < 0.5)) {

        //Saving a sample for after training, making a prediction
        if ((!this.class_0_sample) && (Math.random() < 0.5))

          this.class_0_sample = Object.values(e.xs);

        features.push(Object.values(e.xs));
        target.push(e.ys.HeartDisease);
        counter_class_0++;
      }
      else if ((e.ys.HeartDisease === 1) && (counter_class_1 < number_of_samples) && (Math.random() < 0.5)) {


        if ((!this.class_1_sample) && (Math.random() < 0.5))
          this.class_1_sample = Object.values(e.xs);

        counter_class_1++;
        features.push(Object.values(e.xs));
        target.push(e.ys.HeartDisease);
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



  /**Normalization */
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

}
