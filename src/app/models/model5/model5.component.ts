import { Component } from '@angular/core';

//Plotting related (Tensorflow.js)
import * as tfvis from '@tensorflow/tfjs-vis';

//Tensorflow.js
import * as tf from '@tensorflow/tfjs';

// Boston Housing data constants:
const BASE_URL =
  'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/';


@Component({
  selector: 'app-model5',
  templateUrl: './model5.component.html',
  styleUrls: ['./model5.component.scss']
})
export class Model5Component {

  //Datasets
  trainData!: any;
  testData!: any;
  trainDatatrainTargetData!: any;
  testDataTarget!: any;


  // loadData() {
  //   this.dataset = tf.data.csv(csvUrl, {
  //     columnConfigs: {
  //       label: {
  //         isLabel: true,
  //       },
  //     },
  //   });
  // }

}
