import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { Model1Component } from './models/model1/model1.component';
import { Model2Component } from './models/model2/model2.component';
import { Model3Component } from './models/model3/model3.component';
import { Model4Component } from './models/model4/model4.component';
import { Model5Component } from './models/model5/model5.component';
import { Model6Component } from './models/model6/model6.component';
import { Model7Component } from './models/model7/model7.component';
import { Model8Component } from './models/model8/model8.component';

const routes: Routes = [
  { path: "model1", component: Model1Component },
  { path: "model2", component: Model2Component },
  { path: "model3", component: Model3Component },
  { path: "model4", component: Model4Component },
  { path: "model5", component: Model5Component },
  { path: "model6", component: Model6Component },
  { path: "model7", component: Model7Component },
  { path: "model8", component: Model8Component }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
