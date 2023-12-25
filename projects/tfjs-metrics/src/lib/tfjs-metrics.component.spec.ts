import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TfjsMetricsComponent } from './tfjs-metrics.component';

describe('TfjsMetricsComponent', () => {
  let component: TfjsMetricsComponent;
  let fixture: ComponentFixture<TfjsMetricsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ TfjsMetricsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TfjsMetricsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
