import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';


@Injectable({
  providedIn: 'root'
})
export class BatmanbeltService {

  constructor() { }

  f1Score(){



  }

  /**
   * 
   * @param confusion - the confusion matrix for the trained model
   */
  performance_metrics(confusion: number[]){
     const metrics: any={};

     //precision = tp / (tp + fp); 
     const precision = confusion[0] / (confusion[0] + confusion[1]);

     metrics.precision= precision;

    //  const recall = tp / (tp + fn);
    const recall = confusion[0] / (confusion[0] + confusion[2]);

    metrics.recall = recall;

    const f1 = 2 * (precision * recall) / (precision + recall);

    metrics.f1= f1;

    return metrics;


  }


  /**
   * 
   * @param model - this is the trained model
   * @param features  - this a matrix with features to predict upon
   * @param target - the expected target 
   * @returns 
   */
  confusionMatrix(model: tf.Sequential, features: number[], target: number[]) {

    const predictions: any = [];
 
    for (let elem of features) {

      const aux_2: any = elem;

      const aux = this.predict(model, aux_2);

      let respose;

      // 0.5 as threshold for deciding whether we have diabetes or no
      if (aux < 0.5)
        respose = 0;
      else
        respose = 1;      

      predictions.push(respose)

    }


    // Defining predictions, labels and 
    // numClasses
    const lab = tf.tensor1d(target, 'int32');
    const pred = tf.tensor1d(predictions, 'int32');
    
    const num_Cls = 2;

    // // Calling tf.confusionMatrix() method
    const output = tf.math.confusionMatrix(lab, pred, num_Cls);

    return output.dataSync();

  }


  /**
   * Imp. the features as input are not normalized
   * @param model - the tensorflow.js model
   * @param value the feature values to be predicted
   * @returns the prediction as number
   */
  predict(model:tf.Sequential, value:any){

    const prediction = model.predict(tf.tensor2d([[...value]])) as tf.Tensor;

    const prediction_value = prediction.dataSync()[0];

    return prediction_value;
  }


}
