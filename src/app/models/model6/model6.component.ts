import { Component, OnInit } from '@angular/core';

//TensorFlow.js
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
// import { TfjsMetricsService } from 'projects/tfjs-metrics/src/public-api';
import { TfjsMetricsService } from 'tfjs-metrics';
import { BatmanbeltService } from 'src/app/shared/batmanbelt.service';
import { TfjsBiomodelsService } from 'projects/tfjs-biomodels/src/public-api';




// const csvUrl =
//   'https://docs.google.com/spreadsheets/d/e/2PACX-1vTIscV53ecu0x-apaqh1Sk3ED3qlVdMLxh9AcPdhDPH2VNogn-kKfAP8j9MYKWk7_inouIR9dFMDaUe/pub?gid=0&single=true&output=csv';

//6-feature model
// const csvUrl =
//   'https://docs.google.com/spreadsheets/d/e/2PACX-1vQWn_2YyEy10tw8ONExMhXK7QZDp6Q23rkBEjfrnMGdtil2NzBa3VH4HtvASyIlvxdFBZq13807doAd/pub?gid=0&single=true&output=csv';

//1-feature model
// const csvUrl =
//   'https://docs.google.com/spreadsheets/d/e/2PACX-1vTIscV53ecu0x-apaqh1Sk3ED3qlVdMLxh9AcPdhDPH2VNogn-kKfAP8j9MYKWk7_inouIR9dFMDaUe/pub?gid=0&single=true&output=csv';



  //3-feature model
const csvUrl =
'https://docs.google.com/spreadsheets/d/e/2PACX-1vSbZKM90cRA1GzO8e_EwbtjebWHRQ56bF5WTG7Nv6W0PL1GBanw8Tbszb7jS18oeuklk3oLxDWCzza2/pub?gid=0&single=true&output=csv';


@Component({
  selector: 'app-model6',
  templateUrl: './model6.component.html',
  styleUrls: ['./model6.component.scss']
})
export class Model6Component implements OnInit {

  dataset!: any;


  constructor(private readonly batmanbelt: BatmanbeltService, private readonly tfmetrics: TfjsMetricsService, private readonly biomodels: TfjsBiomodelsService){}

  ngOnInit(): void {

    this.loadData();
    this.visualizeDataset();
    this.defineAndTrainModel();

    // console.log(this.tfmetrics.calculator(2,2));

    // this.biomodels.diabetes_model([80, 0, 1, 25.19, 6.6, 140]);

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

    const numberEpochs = 3;

    // numOfFeatures is the number of column or features minus the label column
    const numOfFeatures = (await this.dataset.columnNames()).length - 1;
    const trainingSurface = { name: 'Loss and MSE', tab: 'Training' };

    // const flattenedDataset = this.dataset
    //   .map((e: any) => ({ xs: Object.values(e.xs), ys: Object.values(e.ys) }))
    //   // Convert the features (xs) and labels (ys) to an array
    //   .batch(100)
    //   .shuffle(100, 17); // buffer size and seed

    // Define the model.

    const features: any = [];
    const target: any = [];

    const features_test: any = [];
    const target_test: any = [];
    
    let test_counter_diabetes =0;
    let test_counter_non_diabetes =0;



    let counter_no_diabetes = 0;
    let counter_diabetes = 0;
    
    await this.dataset.shuffle(3);

    await this.dataset.forEachAsync((e: any) => {

      // Extract the features from the dataset
      const aux = { x: e.xs.HbA1c_level, y: e.xs.heart_disease };
      // If the label is 0, add the features to the "classZero" array
      
      if((test_counter_diabetes<50) && (e.ys.diabetes===1)) {//will sample 100 for testing
        features_test.push(Object.values(e.xs));
        target_test.push(e.ys.diabetes);
        test_counter_diabetes++;
      } else if((test_counter_non_diabetes<50) && (e.ys.diabetes===0)){
        features_test.push(Object.values(e.xs));
        target_test.push(e.ys.diabetes);
        
        test_counter_non_diabetes++;
      } else  {//we already have enough testing samples
        features.push(Object.values(e.xs));
        target.push(e.ys.diabetes);  
      }    


    });

    this.shuffle(features, target);

    const features_tensor_raw = tf.tensor2d(features, [features.length, numOfFeatures]);
    const target_tensor = tf.tensor2d(target, [target.length, 1]);    


    //mean and std
    // let { dataMean, dataStd } = this.determineMeanAndStddev(features_tensor_raw);
    // const features_tensor = this.normalizeTensor(features_tensor_raw, dataMean, dataStd);

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

    // // Output value should be near 0.
    // (model.predict(tf.tensor2d([[80, 0, 1, 25.19, 6.6, 140]])) as tf.Tensor).print();
    // // Output value should be near 1.
    // (model.predict(tf.tensor2d([[44, 0, 0, 19.31, 6.5, 200]])) as tf.Tensor).print();

    // const aux_prediction = this.batmanbelt.predict(model,[44, 0, 0, 19.31, 6.5, 200]);
    
    // console.log("Output value should be near 1: ", aux_prediction);


    //Downloading the final model    
    await model.save('downloads://my-model');
    console.log(target_test);
    
    // const confusion_matrix: any =  await this.batmanbelt.confusionMatrix(model, features_test, target_test);
    const confusion_matrix: any =  await this.tfmetrics.confusionMatrix(model, features_test, target_test);

    console.log([confusion_matrix]);

    // const performance= this.batmanbelt.performance_metrics(confusion_matrix);
    const performance= this.tfmetrics.performance_metrics(confusion_matrix);

    console.log("Performance: ", performance);


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

