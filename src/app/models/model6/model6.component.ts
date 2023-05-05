import { Component, OnInit } from '@angular/core';

//TensorFlow.js
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

const csvUrl =
  'https://docs.google.com/spreadsheets/d/e/2PACX-1vTIscV53ecu0x-apaqh1Sk3ED3qlVdMLxh9AcPdhDPH2VNogn-kKfAP8j9MYKWk7_inouIR9dFMDaUe/pub?gid=0&single=true&output=csv';


@Component({
  selector: 'app-model6',
  templateUrl: './model6.component.html',
  styleUrls: ['./model6.component.scss']
})
export class Model6Component implements OnInit {

  dataset!: any;

  ngOnInit(): void {
    this.loadData();
    this.visualizeDataset();
    this.defineAndTrainModel();
  }



  loadData() {
    // Our target variable (what we want to predict) is the column 'label' (wow, very original),
    // so we specify it in the configuration object as the label
    this.dataset = tf.data.csv(csvUrl, {
      columnConfigs: {
        diabetes: {
          isLabel: true,
        },
      },
    });
  }

  async visualizeDataset() {
    // tfjs-vis surface's names and tabs
    const dataSurface = { name: 'Diabetes prediction dataset (Kaggle)', tab: 'Charts' };
    const classZero: any = [];
    const classOne: any = [];


    await this.dataset.forEachAsync((e: any) => {
      // Extract the features from the dataset
      const features = { x: e.xs.HbA1c_level, y: e.xs.heart_disease };
      // If the label is 0, add the features to the "classZero" array
      if (e.ys.diabetes === 0) {
        classZero.push(features);
      } else {
        classOne.push(features);
      }
    });

    // Specify the name of the labels. This will show up in the chart's legend.
    const series = ['No diabetes', 'Diabetes'];
    const dataToDraw = { values: [classZero, classOne], series };

    tfvis.render.scatterplot(dataSurface, dataToDraw, {
      xLabel: 'Body Mass Index',
      yLabel: 'Blood Glucose Level',
      zoomToFit: true,
    });
  }


  async defineAndTrainModel() {

    const numberEpochs = 300;

    // numOfFeatures is the number of column or features minus the label column
    const numOfFeatures = (await this.dataset.columnNames()).length - 1;
    const trainingSurface = { name: 'Loss and MSE', tab: 'Training' };

    const flattenedDataset = this.dataset
      .map((e: any) => ({ xs: Object.values(e.xs), ys: Object.values(e.ys) }))
      // Convert the features (xs) and labels (ys) to an array
      .batch(100)
      .shuffle(100, 17); // buffer size and seed

    // Define the model.

    const features: any = [];
    const target: any = [];

    let counter_no_diabetes = 0;
    let counter_diabetes = 0;

    await this.dataset.forEachAsync((e: any) => {
      // Extract the features from the dataset
      const aux = { x: e.xs.HbA1c_level, y: e.xs.heart_disease };
      // If the label is 0, add the features to the "classZero" array
      features.push(Object.values(e.xs));
      target.push(e.ys.diabetes);

      if (e.ys.diabetes === 0)
        counter_no_diabetes++;
      else
        counter_diabetes++
    });

    console.log("No diabetes", counter_no_diabetes);
    console.log("diabetes", counter_diabetes);

    this.shuffle(features, target);

    const features_tensor_raw = tf.tensor2d(features, [features.length, numOfFeatures]);
    const target_tensor = tf.tensor2d(target, [target.length, 1])

    // features_tensor.print();
    // features_tensor.print();
    // const aux = this.normalize(features_tensor);

    //mean and std
    let { dataMean, dataStd } = this.determineMeanAndStddev(features_tensor_raw);
    const features_tensor = this.normalizeTensor(features_tensor_raw, dataMean, dataStd);

    // aux.print()

    // const trainTensors = {
    //   sizeMB: tf.tensor2d(this.trainData.sizeMB, [20, 1]),
    //   timeSec: tf.tensor2d(this.trainData.timeSec, [20, 1])
    // };

    // console.log(features);
    // console.log(target);

    const model = tf.sequential();

    // Add a Dense layer to the Sequential model

    //Input layer
    model.add(
      tf.layers.dense({
        inputShape: [numOfFeatures],
        units: 50,
        activation: 'relu',
      })
    );

    //Hidden Layers

    // model.add(
    //   tf.layers.dense({
    //     inputShape: [numOfFeatures],
    //     units: 512,
    //     activation: 'relu',
    //   })
    // );

    // model.add(
    //   tf.layers.dense({
    //     inputShape: [numOfFeatures],
    //     units: 256,
    //     activation: 'relu',
    //   })
    // );


    // model.add(
    //   tf.layers.dense({
    //     inputShape: [numOfFeatures],
    //     units: 128,
    //     activation: 'relu',
    //   })
    // );

    // model.add(
    //   tf.layers.dense({
    //     inputShape: [numOfFeatures],
    //     units: 56,
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

    // Output value should be near 0.
    (model.predict(tf.tensor2d([[80, 0, 1, 25.19, 6.6, 140]])) as tf.Tensor).print();
    // Output value should be near 1.
    (model.predict(tf.tensor2d([[44, 0, 0, 19.31, 6.5, 200]])) as tf.Tensor).print();

  }


  normalize(data: tf.Tensor) {
    const dataMax = data.max();
    const dataMin = data.min();
    return data.sub(dataMin).div(dataMax.sub(dataMin));
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

  normalizeTensor(data: tf.Tensor, dataMean: tf.Tensor, dataStd: tf.Tensor) {
    return data.sub(dataMean).div(dataStd);
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

  // standardize(data: tf.Tensor) {
  //   let means = []
  //   let variances: any = []
  //   for (let axe = 0; axe < axes; axe++) {
  //     const { mean, variance } = tf.moments(data.gather([axe], 1), undefined, true)
  //     means.push(mean)
  //     variances.push(variances)
  //   }
  //   return data.sub(tf.concat(means).reshape([axes])).div(tf.concat(variances).reshape([axes]))
  // }


}

