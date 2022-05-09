import { Component, OnInit } from '@angular/core';
import { ModalController, NavParams } from '@ionic/angular';

@Component({
  selector: 'ksky-sentiment-modal',
  templateUrl: './sentiment-modal.component.html',
  styleUrls: ['./sentiment-modal.component.scss'],
})
export class SentimentModalComponent implements OnInit {

  constructor(
    private modalController: ModalController,
    private navParams: NavParams
  ) { }

  ngOnInit() {}

  dismiss() {
    this.modalController.dismiss();
  }

}
