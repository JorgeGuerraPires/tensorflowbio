import { Component, OnInit } from '@angular/core';

//TensorFlow.js
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';


const csvUrl =
  'https://docs.google.com/spreadsheets/d/e/2PACX-1vRDvX6wlrHbhrp4vO2oGjzui8Rjk6Wtybgo_MFDCVhNiuYDdy8DSu_TiK2GGYwE62sypEez3Z7gpjAZ/pub?gid=0&single=true&output=csv'


@Component({
  selector: 'app-model3',
  templateUrl: './model3.component.html',
  styleUrls: ['./model3.component.scss']
})
export class Model3Component implements OnInit {
  dataset: any;

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
        token: {
          isLabel: true,
        },
      },
    });
  }

  async visualizeDataset() {
    // tfjs-vis surface's names and tabs
    const dataSurface = { name: 'Scatterplot', tab: 'Charts' };
    const data: any = [];


    await this.dataset.forEachAsync((e: any) => {
      // Extract the features from the dataset
      const aux = { x: e.xs.palavras, y: e.ys.token };
      data.push(aux);
    });

    // Specify the name of the labels. This will show up in the chart's legend.
    const series = ['dataset'];
    const dataToDraw = { values: data, series };

    tfvis.render.scatterplot(dataSurface, dataToDraw, {
      xLabel: 'palavras',
      yLabel: 'token',
      zoomToFit: true,
    });
  }

  async defineAndTrainModel() {
    // numOfFeatures is the number of column or features minus the label column
    const numOfFeatures = (await this.dataset.columnNames()).length - 1;
    const trainingSurface = { name: 'Loss and MSE', tab: 'Training' };

    const flattenedDataset = this.dataset
      .map((e: any) => ({ xs: Object.values(e.xs), ys: Object.values(e.ys) }))

      // Convert the features (xs) and labels (ys) to an array
      .batch(10)
    // .shuffle(100, 17); // buffer size and seed

    // Define the model.
    const model = tf.sequential();

    // Add a Dense layer to the Sequential model
    model.add(tf.layers.dense({ inputShape: [1], units: 1 }));
    // model.add(
    //   tf.layers.dense({
    //     inputShape: [numOfFeatures],
    //     units: 1,
    //     activation: 'sigmoid',
    //   })
    // );

    model.compile({ optimizer: tf.train.adam(0.1), loss: 'meanAbsoluteError' });


    // model.compile({
    //   optimizer: tf.train.adam(0.1),
    //   loss: 'binaryCrossentropy',
    //   metrics: ['accuracy'],
    // });

    // Print the summary to console
    model.summary();

    // console.log(flattenedDataset)
    // Fit the model
    await model.fitDataset(flattenedDataset, {
      epochs: 100,
      callbacks: [
        // Show on a tfjs-vis visor the loss and accuracy values at the end of each epoch.
        tfvis.show.fitCallbacks(trainingSurface, ['loss', 'acc'], {
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
    (model.predict(tf.tensor1d([786])) as tf.Tensor).print();
    // // Output value should be near 1.
    // (model.predict(tf.tensor2d([[51, 32.79]])) as tf.Tensor).print();
  }

}


