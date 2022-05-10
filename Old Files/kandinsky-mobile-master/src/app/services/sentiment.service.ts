import { HttpClient } from '@angular/common/http';
import { Injectable, OnInit } from '@angular/core';
import { Observable } from 'rxjs';
import { Sentiment } from '../sentiment.model';

@Injectable()
export class SentimentService {
  constructor(private http: HttpClient) {}

  ngOnInit(){
  }

  // { sentiment: 'POSITIVE' }
  // return this.http.post<Sentiment>('', { comment });

  sentimentURL : string = "http://127.0.0.1:5000/sentiment"

  getSentiment(): Observable<Sentiment> {
    return this.http.get<Sentiment>(this.sentimentURL);
  }
}
