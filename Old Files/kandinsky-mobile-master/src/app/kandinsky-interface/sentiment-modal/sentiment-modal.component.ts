import { Component, OnInit } from '@angular/core';
import { ModalController, NavParams } from '@ionic/angular';
import { CommentSentiment } from 'src/app/models/models';

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
  ) { }

  ngOnInit() {
    // this.comment = this.navParams.get('comment');
  }

  dismiss() {
    this.modalController.dismiss();
  }

}
