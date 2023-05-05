import { Component, OnInit } from '@angular/core';

//TensorFlow.js
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';


@Component({
  selector: 'app-model4',
  templateUrl: './model4.component.html',
  styleUrls: ['./model4.component.scss']
})
export class Model4Component implements OnInit {

  //Datasets
  trainData = {
    hydrocarbon_level: [1.02, 1.15, 1.29, 1.46, 1.36, 0.87, 1.23, 1.55, 1.40, 1.19, 1.15, 0.98],
    purity: [89.05, 91.43, 93.74, 96.73, 94.45, 87.59, 91.77, 99.42, 93.65, 93.54, 92.52, 90.56]
  };
  testData = {
    hydrocarbon_level: [1.20, 1.26, 1.32, 1.43, 0.95, 1.11, 1.01, 0.99],
    purity: [90.39, 93.25, 93.41, 94.98, 87.33, 89.85, 89.54, 90.01]
  };

  ngOnInit(): void {
    this.plotting();
    this.linearregression();
  }

  //plotting
  plotting() {
    // const lossContainer = document.getElementById('traindataset');

    //the plotting requires the format vector=[{a,b},{c,d},,]
    //Train dataset
    const traindataset = this.trainData.hydrocarbon_level.map((elem, i) => { return { x: elem, y: this.trainData.purity[i] } });
    //test dataset
    const testdataset = this.testData.hydrocarbon_level.map((elem, i) => { return { x: elem, y: this.testData.purity[i] } });

    //plotting the scatter plot
    tfvis.render.scatterplot(
      { name: 'Scatterplot', tab: 'Charts' },
      {
        values: [traindataset, testdataset], series: ['train dataset', 'test dataset']
      },
      {
        width: 420,
        height: 300,
        xLabel: 'Hydrocarbon Level (%)',
        yLabel: 'Purity  (%)',
        xAxisDomain: [0.5, 2],
        yAxisDomain: [85, 100]
      }
    );

  }


  linearregression() {

    //Creating the tensors
    //Data used to train
    const trainTensors = {
      sizeMB: tf.tensor2d(this.trainData.hydrocarbon_level, [this.trainData.hydrocarbon_level.length, 1]),
      timeSec: tf.tensor2d(this.trainData.purity, [this.trainData.purity.length, 1])
    };

    //Data used to test
    const testTensors = {
      sizeMB: tf.tensor2d(this.testData.hydrocarbon_level, [this.testData.hydrocarbon_level.length, 1]),
      timeSec: tf.tensor2d(this.testData.purity, [this.testData.purity.length, 1])
    };


    //Model building
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [1], units: 1 }));

    //Training
    model.compile({ optimizer: 'sgd', loss: 'meanAbsoluteError' });

    model.fit(trainTensors.sizeMB,
      trainTensors.timeSec,
      {
        epochs: 10000,
        callbacks: [
          // Show on a tfjs-vis visor the loss and accuracy values at the end of each epoch.
          tfvis.show.fitCallbacks({ name: 'Loss and MSE', tab: 'Training' }, ['loss', 'acc'], {
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
      }).then(() => {

        //calculating some simple metric for comparison
        // this.suporte();

        // document.getElementById('neural network').innerText =
        //   (model.evaluate(testTensors.sizeMB, testTensors.timeSec) as tf.Tensor).toString().split("\n")[1];

      });



  }




}
