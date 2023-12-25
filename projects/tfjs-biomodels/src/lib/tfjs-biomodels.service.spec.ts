import { TestBed } from '@angular/core/testing';

import { TfjsBiomodelsService } from './tfjs-biomodels.service';

describe('TfjsBiomodelsService', () => {
  let service: TfjsBiomodelsService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(TfjsBiomodelsService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
