import { Component, OnInit } from '@angular/core';
import { ModalController, NavParams } from '@ionic/angular';
import { CommentSentiment } from 'src/app/models/models';
import { AnalyticsService } from 'src/app/services/analytics.service';


@Component({
  selector: 'ksky-sentiment-modal',
  templateUrl: './sentiment-modal.component.html',
  styleUrls: ['./sentiment-modal.component.scss'],
})
export class SentimentModalComponent implements OnInit {

  


  comment: CommentSentiment;

  constructor(
    private modalController: ModalController,
    private navParams: NavParams
    , public analyticsSrv:AnalyticsService
  ) { }

  ngOnInit() {
    // this.comment = this.navParams.get('comment');
   // this.analyticsSrv.ReturnedData.senticnet1
  }

  dismiss() {
    this.modalController.dismiss();
  }

}