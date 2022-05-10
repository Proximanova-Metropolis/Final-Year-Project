import { Injectable } from '@angular/core';
import * as lda from 'lda';
import * as similarity from 'compute-cosine-similarity';
import _ from 'lodash';
import { getIndices } from '../utils';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { AnalyzedResult } from '../models/models';


@Injectable({
  providedIn: 'root'
})
export class AnalyticsService {
 public CheckedStatus={ sn:true, swn:true,vd:true  }

 public ReturnedData:AnalyzedResult;

 

  constructor(public httpClient: HttpClient) {
  }

  public generateTopics(text: string, terms: number = 10): {[k: string]: Topic} {
    var sentences = text.match(/[^\.!\?]+[\.!\?]+/g);
    const topics: { term: string, probability: number }[] = lda(sentences, 1, terms)[0] || [];
    
    const topicMap = {};
    topics.forEach(topic => {
      const regex = new RegExp(_.escapeRegExp(topic.term), 'gi');

      topicMap[topic.term] = {
        term: topic.term,
        indices: getIndices(regex, text),
        probability: topic.probability
      };
    })

    return topicMap;
  }

  public calculateTopicSimilarity(a: {[k: string]: Topic}, b: {[k: string]: Topic}) {

    const terms = new Set([...Object.keys(a), ...Object.keys(b)]);
    const aVector = [...terms].map(t => t in a ? a[t].probability : 0);
    const bVector = [...terms].map(t => t in b ? b[t].probability : 0);
    
    return similarity(aVector, bVector);
  }


  public AnalyzeComment(ana_text:string ){
   let pr_sn= this.CheckedStatus.sn?1:0;
   let pr_swn= this.CheckedStatus.swn?1:0;
  let  pr_vd= this.CheckedStatus.vd?1:0;
   return this.httpClient.post('http://127.0.0.1:7775/analysys',{ text:ana_text,sn:pr_sn,swn:pr_swn,vd:pr_vd})

  }


}

export type Topic = {
  term: string;
  indices: [number, number][];
  probability: number;
}