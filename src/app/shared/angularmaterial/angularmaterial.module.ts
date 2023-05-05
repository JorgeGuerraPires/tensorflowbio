import { NgModule } from '@angular/core';


import { MatInputModule } from '@angular/material/input';
import { MatSliderModule } from '@angular/material/slider';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';


const angularmaterialapis = [MatInputModule, MatSliderModule, MatButtonModule, MatCardModule]

@NgModule({
  declarations: [],
  imports: [...angularmaterialapis],
  exports: [...angularmaterialapis]
})
export class AngularmaterialModule { }
