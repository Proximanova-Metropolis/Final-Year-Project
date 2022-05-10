import { Component } from '@angular/core';
import { Platform } from '@ionic/angular';
import { SplashScreen } from '@ionic-native/splash-screen/ngx';
import { StatusBar } from '@ionic-native/status-bar/ngx';
import { SentimentService } from './services/sentiment.service';
import { Sentiment } from './sentiment.model';

@Component({
  selector: 'app-root',
  templateUrl: 'app.component.html',
  styleUrls: ['app.component.scss']
})
export class AppComponent {
  constructor(
    private platform: Platform,
    private splashScreen: SplashScreen,
    private statusBar: StatusBar,
    private rs: SentimentService
  ) {
    this.initializeApp();
  }

  sentiment: Sentiment;

  initializeApp() {
    this.platform.ready().then(() => {
    this.statusBar.styleDefault();
    this.splashScreen.hide();
    this.rs.getSentiment().subscribe(
      (response) =>
      {
        this.sentiment = response;
      },
      (error) =>
      {
        console.log("No Data Found" + error);
      }
    )
    });
  }
}
