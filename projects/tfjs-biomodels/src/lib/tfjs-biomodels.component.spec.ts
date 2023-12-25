import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TfjsBiomodelsComponent } from './tfjs-biomodels.component';

describe('TfjsBiomodelsComponent', () => {
  let component: TfjsBiomodelsComponent;
  let fixture: ComponentFixture<TfjsBiomodelsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ TfjsBiomodelsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TfjsBiomodelsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
