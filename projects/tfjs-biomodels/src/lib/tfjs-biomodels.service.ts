import { Injectable } from '@angular/core';

//TensorFlow.js
import * as tf from '@tensorflow/tfjs';


@Injectable({
  providedIn: 'root'
})
export class TfjsBiomodelsService {

  constructor() { }

  async diabetes_model(features: number[]){

    console.log("I am on diabetes model...");

    //Location of the model, should be downloaded
    
    const modelPath = 'assets/models/6-feature-model.json';

    let output: any = {message: "no prediction done"};

    return tf.loadLayersModel(modelPath).then((model) => {

          const input = tf.tensor2d([features]);

          const result =  model.predict(input) as tf.Tensor;

          console.log("Predicting");

          result.print(); 

          const prediction_value = result.dataSync()[0];

          console.log("Prediction: ", prediction_value)

          output= {probability: prediction_value};


          return JSON.stringify(output);
        
        })


  }//end of diabetes model

}
