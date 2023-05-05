import { Component } from '@angular/core';

//TensorFlow.js
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';


@Component({
  selector: 'app-model1',
  templateUrl: './model1.component.html',
  styleUrls: ['./model1.component.scss']
})
export class Model1Component {

  //Neural Network parametrization
  NNepochs = 2;
  working = false;

  longText = `Use this card to parametrize your model`;

  //
  evaluation!: string;

  //Datasets
  trainData = {
    sizeMB: [0.080, 9.000, 0.001, 0.100, 8.000,
      5.000, 0.100, 6.000, 0.050, 0.500,
      0.002, 2.000, 0.005, 10.00, 0.010,
      7.000, 6.000, 5.000, 1.000, 1.000],
    timeSec: [0.135, 0.739, 0.067, 0.126, 0.646,
      0.435, 0.069, 0.497, 0.068, 0.116,
      0.070, 0.289, 0.076, 0.744, 0.083,
      0.560, 0.480, 0.399, 0.153, 0.149]
  };
  testData = {
    sizeMB: [5.000, 0.200, 0.001, 9.000, 0.002,
      0.020, 0.008, 4.000, 0.001, 1.000,
      0.005, 0.080, 0.800, 0.200, 0.050,
      7.000, 0.005, 0.002, 8.000, 0.008],
    timeSec: [0.425, 0.098, 0.052, 0.686, 0.066,
      0.078, 0.070, 0.375, 0.058, 0.136,
      0.052, 0.063, 0.183, 0.087, 0.066,
      0.558, 0.066, 0.068, 0.610, 0.057]
  };

  model: tf.Sequential = tf.sequential();




  linearregression(el: HTMLElement) {
    this.working = true;

    el.scrollIntoView();

    //Pre-preparation
    //Creating the tensors
    const trainTensors = {
      sizeMB: tf.tensor2d(this.trainData.sizeMB, [20, 1]),
      timeSec: tf.tensor2d(this.trainData.timeSec, [20, 1])
    };

    const testTensors = {
      sizeMB: tf.tensor2d(this.testData.sizeMB, [20, 1]),
      timeSec: tf.tensor2d(this.testData.timeSec, [20, 1])
    };

    //Building the model
    this.model = tf.sequential();

    this.model.add(tf.layers.dense({ inputShape: [1], units: 1 }));

    //Training 
    this.model.compile({ optimizer: 'sgd', loss: 'meanAbsoluteError' });

    this.model.fit(trainTensors.sizeMB,
      trainTensors.timeSec,
      { epochs: this.NNepochs }).then(() => {

        // console.log("Network prediction");

        //calculating some simple metric for comparison
        this.suporte();
        this.evaluation = (this.model.evaluate(testTensors.sizeMB, testTensors.timeSec) as tf.Tensor).toString().split("\n")[1];

        this.working = false

      });

  }

  //plotting
  plotting() {
    //Train dataset
    const traindataset = this.trainData.sizeMB.map((elem, i) => { return { x: elem, y: this.trainData.timeSec[i] } })

    //test dataset
    const testdataset = this.testData.sizeMB.map((elem, i) => { return { x: elem, y: this.testData.timeSec[i] } })


    tfvis.render.scatterplot(
      { name: 'Scatterplot', tab: 'Charts' },
      {
        values: [traindataset, testdataset], series: ['train dataset', 'test dataset']
      },
      {
        width: 420,
        height: 300,
        xLabel: 'file size (MB)',
        yLabel: 'Time to download (sec)',
      }
    );

  }

  average!: string;
  abs!: string;

  suporte() {
    console.log("Our guessing: ");

    const avgDelaySec = tf.mean(this.trainData.timeSec);
    this.average = avgDelaySec.toString().split("\n")[1];

    this.abs = tf.mean(tf.abs(tf.sub(this.testData.timeSec, 0.295))).toString().split("\n")[1];

  }

}
