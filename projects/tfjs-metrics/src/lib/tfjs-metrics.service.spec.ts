import { TestBed } from '@angular/core/testing';

import { TfjsMetricsService } from './tfjs-metrics.service';

describe('TfjsMetricsService', () => {
  let service: TfjsMetricsService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(TfjsMetricsService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
