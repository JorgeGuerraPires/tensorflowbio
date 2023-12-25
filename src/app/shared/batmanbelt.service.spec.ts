import { TestBed } from '@angular/core/testing';

import { BatmanbeltService } from './batmanbelt.service';

describe('BatmanbeltService', () => {
  let service: BatmanbeltService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(BatmanbeltService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
